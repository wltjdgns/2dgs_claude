import math
import torch
import torch.nn.functional as F

from utils.planar_utils import compute_virtual_camera_reflected


def _reflection_warp(img_virtual, virtual_cam, xyz_map, normal_map, plane_mask, cam_center):
    """
    Per-pixel reflection re-projection.

    각 태블릿 픽셀 (u,v)에서:
      1. 3D point P, 법선 N, 카메라 방향 v_dir 계산
      2. 반사 방향 d_refl = v_dir - 2*(v_dir·N)*N
      3. d_refl을 가상 카메라 공간으로 변환 → 픽셀 좌표 (u', v')
      4. img_virtual에서 (u', v') grid_sample → 원본 (u,v)에 합성

    Args:
        img_virtual : (3, H, W) 가상 카메라 렌더
        virtual_cam : 가상 카메라 객체 (world_view_transform, FoVx, FoVy)
        xyz_map     : (3, H, W) world-space 3D positions
        normal_map  : (3, H, W) world-space normals
        plane_mask  : (H, W) bool — 태블릿 영역
        cam_center  : (3,) 원본 카메라 위치
    Returns:
        warped: (3, H, W) — plane_mask 영역에만 반사 색상
    """
    device = img_virtual.device
    _, H, W = img_virtual.shape

    P     = xyz_map.permute(1, 2, 0)                           # (H, W, 3)
    N     = F.normalize(normal_map.permute(1, 2, 0), dim=-1)   # (H, W, 3)
    v_dir = F.normalize(cam_center.view(1, 1, 3) - P, dim=-1)  # (H, W, 3)

    # 반사 방향: d_refl = v_dir - 2*(v_dir·N)*N
    dot_vn = (v_dir * N).sum(dim=-1, keepdim=True)             # (H, W, 1)
    d_refl = F.normalize(v_dir - 2 * dot_vn * N, dim=-1)       # (H, W, 3)

    # world → 가상 카메라 좌표 변환 (방향 벡터)
    # world_view_transform[:3,:3] = C2W_R → .T = W2C_R
    W2C_R = virtual_cam.world_view_transform[:3, :3].T          # (3, 3)
    d_cam = (W2C_R @ d_refl.reshape(-1, 3).T).T.reshape(H, W, 3)  # (H, W, 3)

    # Pinhole projection → 픽셀 좌표 (u', v')
    focal_x = W / (2.0 * math.tan(virtual_cam.FoVx / 2.0))
    focal_y = H / (2.0 * math.tan(virtual_cam.FoVy / 2.0))

    d_z     = d_cam[..., 2].clamp(min=1e-6)
    u_prime = d_cam[..., 0] / d_z * focal_x + W / 2.0          # (H, W)
    v_prime = d_cam[..., 1] / d_z * focal_y + H / 2.0          # (H, W)

    # NDC [-1, 1] for grid_sample
    u_ndc = (u_prime / (W - 1)) * 2.0 - 1.0
    v_ndc = (v_prime / (H - 1)) * 2.0 - 1.0
    grid  = torch.stack([u_ndc, v_ndc], dim=-1).unsqueeze(0)   # (1, H, W, 2)

    # 가상 카메라 이미지에서 반사 방향에 대응하는 픽셀 샘플링
    warped = F.grid_sample(
        img_virtual.unsqueeze(0), grid,
        mode='bilinear', padding_mode='zeros', align_corners=False,
    ).squeeze(0)  # (3, H, W)

    # 가상 카메라 뒤를 향하는 반사 방향 (d_cam.z <= 0) 및 마스크 외부 제거
    valid = (d_cam[..., 2] > 0) & plane_mask                   # (H, W)
    return warped * valid.float().unsqueeze(0)

try:
    from utils import brdf_utils as brdf
except Exception:
    brdf = None


def _reconstruct_xyz_map(viewpoint_camera, surf_depth):
    """
    Reconstruct per-pixel world-space positions from surf_depth (1, H, W).

    2DGS does not output xyz_map directly, so we backproject using camera
    intrinsics and the world-view transform inverse.

    Returns:
        xyz_map: (3, H, W) world-space positions, zeros where depth is invalid
    """
    device = surf_depth.device
    depth = surf_depth[0]  # (H, W)
    H, W = depth.shape

    focal_x = viewpoint_camera.image_width  / (2 * math.tan(viewpoint_camera.FoVx / 2))
    focal_y = viewpoint_camera.image_height / (2 * math.tan(viewpoint_camera.FoVy / 2))

    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    # Assumes principal point at image center (W/2, H/2).
    # For datasets with off-center principal point, supply cx/cy from camera params.
    x_cam = (grid_x - W / 2) / focal_x * depth
    y_cam = (grid_y - H / 2) / focal_y * depth
    ones  = torch.ones_like(depth)

    pts_cam = torch.stack([x_cam, y_cam, depth, ones], dim=0)  # (4, H, W)
    c2w = torch.inverse(viewpoint_camera.world_view_transform.T)  # C2W = W2C^{-1}
    pts_world = (c2w @ pts_cam.reshape(4, -1)).reshape(4, H, W)

    xyz_map = pts_world[:3]  # (3, H, W)
    xyz_map = xyz_map * (depth > 0.01).unsqueeze(0).float()
    return xyz_map


def render_2pass(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    render_func,
    scaling_modifier=1.0,
    override_color=None,
    lambda_weight=0.5,
    enable_2pass=True,
    iteration=0,
    planar_groups=None,
    metal_map=None,
    render_pkg_base=None,
):
    """
    Two-pass rendering with planar reflection and BRDF-style composition.

    2DGS adaptation:
      - Uses 'surf_depth' and 'rend_normal' from render_func output
      - Reconstructs xyz_map from surf_depth via backprojection
      - 'rend_normal' is the 2DGS surfel normal in world space

    Args:
        planar_groups:   List[Dict] from detect_planar_groups_from_depth_fast.
                         If None, 2-pass is disabled (fallback to base rendering).
        render_pkg_base: Pre-computed base render package (e.g. with f0_map already
                         injected from MetalicNet). If None, base render is run internally.
    """
    device = pc.get_xyz.device

    # ===== Step 1: Base rendering (skip if pre-computed result provided) =====
    if render_pkg_base is None:
        render_pkg_base = render_func(
            viewpoint_camera, pc, pipe, bg_color,
            scaling_modifier=scaling_modifier,
            override_color=override_color,
        )

    img_base = render_pkg_base["render"]
    H, W = img_base.shape[1], img_base.shape[2]

    f0_map = render_pkg_base.get("f0_map", None)

    # ===== Early exit if 2-pass disabled =====
    if not enable_2pass or planar_groups is None or len(planar_groups) == 0:
        return {
            "render":            img_base,
            "viewspace_points":  render_pkg_base["viewspace_points"],
            "visibility_filter": render_pkg_base["visibility_filter"],
            "radii":             render_pkg_base["radii"],
            "rend_dist":         render_pkg_base.get("rend_dist",   None),
            "rend_normal":       render_pkg_base.get("rend_normal", None),
            "surf_normal":       render_pkg_base.get("surf_normal", None),
        }

    # ===== Step 2: Multi-plane specular accumulation =====
    img_specular_accum = torch.zeros_like(img_base)
    specular_mask_accum = torch.zeros(1, H, W, device=device)
    plane_results = []

    # xyz_map은 모든 plane에서 공통 — loop 밖에서 한 번만 계산
    _surf_depth_pre  = render_pkg_base.get("surf_depth",  None)
    _normal_map_pre  = render_pkg_base.get("rend_normal", None)
    _xyz_map_pre     = _reconstruct_xyz_map(viewpoint_camera, _surf_depth_pre) if _surf_depth_pre is not None else None

    for group in planar_groups:
        plane_mask   = group["mask"]           # (H, W)
        plane_normal = group["dominant_normal"] # (3,)
        plane_center = group["center"]          # (3,)

        cam_to_plane = plane_center - viewpoint_camera.camera_center
        if torch.dot(cam_to_plane, plane_normal) < 0:
            plane_normal = -plane_normal

        virtual_camera = compute_virtual_camera_reflected(
            viewpoint_camera, plane_normal, plane_center
        )
        render_pkg_virtual = render_func(
            virtual_camera, pc, pipe, bg_color,
            scaling_modifier=scaling_modifier,
        )
        img_virtual   = render_pkg_virtual["render"]
        plane_mask_3d = plane_mask.unsqueeze(0).float()  # (1, H, W)

        # Per-pixel reflection re-projection (정확한 warp)
        if _xyz_map_pre is not None and _normal_map_pre is not None:
            warped = _reflection_warp(
                img_virtual, virtual_camera,
                _xyz_map_pre, _normal_map_pre,
                plane_mask, viewpoint_camera.camera_center,
            )
        else:
            # fallback: xyz/normal 없으면 기존 단순 crop 방식
            warped = img_virtual * plane_mask_3d

        plane_results.append({
            "mask":     plane_mask_3d,
            "specular": img_virtual,   # raw 가상 카메라 렌더 (저장용)
            "warped":   warped,        # warp된 반사 (합성용)
            "normal":   plane_normal,
            "center":   plane_center,
        })

        img_specular_accum  += warped
        specular_mask_accum += plane_mask_3d

    # Normalize specular by overlap count so overlapping planes don't over-accumulate
    overlap_count = specular_mask_accum.clamp(min=1.0)
    img_specular_accum = img_specular_accum / overlap_count
    specular_mask_accum = specular_mask_accum.clamp(0, 1)

    # ===== BRDF-style composition =====
    # --- roughness map (1, H, W) ---
    roughness_map = render_pkg_base.get("roughnessmap", None)
    if roughness_map is not None and roughness_map.dim() == 2:
        roughness_map = roughness_map.unsqueeze(0)
    r = roughness_map.clamp(0, 1) if roughness_map is not None else torch.zeros(1, H, W, device=device)

    # --- planar / specular mask ---
    M = specular_mask_accum.clamp(0, 1)

    # --- F0 (per-pixel base reflectance) ---
    F0 = f0_map if f0_map is not None else torch.full_like(img_base, 0.04)

    # --- cos(theta_o) = n · v ---
    # 2DGS: use 'rend_normal' (surfel world-space normals)
    normal_map = render_pkg_base.get("rend_normal", None)
    surf_depth  = render_pkg_base.get("surf_depth",  None)

    if normal_map is not None and surf_depth is not None:
        n = F.normalize(normal_map, dim=0)  # (3, H, W)
        xyz_map = _xyz_map_pre  # 이미 계산된 값 재사용
        cam_center = viewpoint_camera.camera_center.view(3, 1, 1)
        v = F.normalize(cam_center - xyz_map, dim=0)
        cos_theta = (n * v).sum(0, keepdim=True).clamp(0, 1)  # (1, H, W)

        # representative half vector
        l = F.normalize(v - 2 * (v * n).sum(0, keepdim=True) * n, dim=0)
        h = F.normalize(v + l, dim=0)
        cos_theta_d = (v * h).sum(0, keepdim=True).clamp(0, 1)
    else:
        cos_theta   = torch.ones(1, H, W, device=device)
        cos_theta_d = cos_theta

    # --- Fresnel, G, D prefilter ---
    if brdf is not None and hasattr(brdf, "fresnel_schlick"):
        F_term = brdf.fresnel_schlick(F0, cos_theta_d)
    else:
        F_term = F0 + (1.0 - F0) * torch.pow(1.0 - cos_theta_d, 5.0)

    if brdf is not None and hasattr(brdf, "g_rep"):
        G_term = brdf.g_rep(r, cos_theta)
    else:
        G_term = torch.ones_like(F_term)

    if brdf is not None and hasattr(brdf, "prefilter_specular"):
        I2_pref = brdf.prefilter_specular(img_specular_accum, r)
    else:
        I2_pref = img_specular_accum

    # --- metalness (from MetalicNet, per-pixel) ---
    if metal_map is not None:
        metal = metal_map.clamp(0, 1)
        if metal.dim() == 2:
            metal = metal.unsqueeze(0)   # (1, H, W)
    else:
        metal = torch.zeros(1, H, W, device=device)

    # --- specular weight: Fresnel × geometry × planar mask × global scale ---
    # lambda_weight is an artistic knob for reflection intensity, not a 50/50 split
    ks_rgb = torch.clamp(F_term * G_term * M * lambda_weight, 0, 1)  # (3, H, W)

    # --- diffuse weight: energy conservation, suppressed on metallic pixels ---
    # metals: kd → 0 (all energy in specular), dielectrics: kd = (1 - ks)
    kd_rgb = (1.0 - metal) * (1.0 - ks_rgb).clamp(0, 1)

    img_final = kd_rgb * img_base + ks_rgb * I2_pref

    return {
        "render":           img_final,
        "pass1":            img_base,
        "pass2":            img_specular_accum,
        "specular_mask":    specular_mask_accum,
        "albedomap":        render_pkg_base.get("albedomap",    None),
        "roughnessmap":     render_pkg_base.get("roughnessmap", None),
        "plane_results":    plane_results,
        "viewspace_points": render_pkg_base["viewspace_points"],
        "visibility_filter": render_pkg_base["visibility_filter"],
        "radii":            render_pkg_base["radii"],
        "surf_depth":       render_pkg_base.get("surf_depth",   None),
        "rend_normal":      render_pkg_base.get("rend_normal",  None),
        "surf_normal":      render_pkg_base.get("surf_normal",  None),
        "rend_dist":        render_pkg_base.get("rend_dist",    None),
        "num_planes":       len(planar_groups),
        "prefiltered_pass2": I2_pref,
        "ks":               ks_rgb,
        "F0":               F0,
        "F_term":           F_term,
        "lambda":           ks_rgb.mean().item(),
    }
