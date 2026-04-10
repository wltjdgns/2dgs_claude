import torch
import torch.nn.functional as F
import numpy as np
import math
import atexit

_planar_stats = {"total": 0, "detected": 0}

@atexit.register
def _print_planar_summary():
    s = _planar_stats
    if s["total"] > 0:
        print(f"[Planar Detection] detected {s['detected']} / {s['total']} views")

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    sam_model_registry = None
    SamAutomaticMaskGenerator = None


def compute_plane_equation(plane_normal, plane_center):
    """평면 방정식 계산: n^T x + d = 0, returns d"""
    return -torch.dot(plane_normal, plane_center)


def compute_virtual_camera(camera, plane_normal, plane_offset):
    """
    Mirror camera across the plane to create virtual camera.

    Args:
        camera: Original Camera object
        plane_normal: (3,) unit normal
        plane_offset: float, plane equation offset (d in n^T x + d = 0)

    Returns:
        virtual_cam: Mirrored Camera object
    """
    from scene.cameras import Camera

    device = camera.camera_center.device
    camera_center = camera.camera_center  # (3,)

    # Mirror camera center: t' = t - 2 * (n^T c + d) * n
    distance = torch.dot(plane_normal, camera_center) + plane_offset
    t_virtual = camera_center - 2 * distance * plane_normal

    # Mirror rotation: I - 2*n*n^T
    R_mirror = torch.eye(3, device=device) - 2 * torch.outer(plane_normal, plane_normal)
    R_world_to_cam = camera.world_view_transform[:3, :3].t()
    R_virtual_to_world = R_mirror @ R_world_to_cam

    virtual_cam = Camera(
        colmap_id=-1,
        R=R_virtual_to_world.detach().cpu().numpy(),
        T=(-R_virtual_to_world.t() @ t_virtual).detach().cpu().numpy(),
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        image=camera.original_image.cpu(),   # (3, H, W) tensor
        gt_alpha_mask=None,
        image_name="virtual",
        uid=-1,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    )
    return virtual_cam


def compute_virtual_camera_simple(camera, plane_normal, plane_center):
    """compute_virtual_camera wrapper using plane_center."""
    plane_offset = compute_plane_equation(plane_normal, plane_center)
    return compute_virtual_camera(camera, plane_normal, plane_offset)


def compute_virtual_camera_reflected(camera, plane_normal, plane_center):
    """
    Create virtual camera at plane center, looking in reflected direction.

    Physics:
        - Camera position = plane center
        - Look direction = reflection of (input_cam → plane_center)

    Args:
        camera: Original Camera object
        plane_normal: (3,) unit normal
        plane_center: (3,) point on plane

    Returns:
        virtual_cam: Camera at plane center looking in reflected direction
    """
    from scene.cameras import Camera

    device = camera.camera_center.device
    plane_normal = F.normalize(plane_normal.to(device), dim=0)
    plane_center = plane_center.to(device)

    d_in = F.normalize(plane_center - camera.camera_center, dim=0)
    d_refl = F.normalize(d_in - 2 * torch.dot(d_in, plane_normal) * plane_normal, dim=0)

    forward = d_refl
    if torch.abs(forward[1]) < 0.9:
        up_world = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    else:
        up_world = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)

    right = F.normalize(torch.cross(forward, up_world), dim=0)
    up = F.normalize(torch.cross(right, forward), dim=0)

    # world → camera: rows = [right, up, -forward]
    R_world_to_cam = torch.stack([right, up, -forward], dim=0)  # (3, 3)
    R_cam_to_world = R_world_to_cam.T
    T = -R_world_to_cam @ plane_center

    virtual_cam = Camera(
        colmap_id=-1,
        R=R_cam_to_world.detach().cpu().numpy(),
        T=T.detach().cpu().numpy(),
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        image=camera.original_image.cpu(),   # (3, H, W) tensor
        gt_alpha_mask=None,
        image_name="virtual_reflected",
        uid=-1,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    )
    return virtual_cam


def select_gaussians_by_plane(
    gaussians,
    plane_normal: torch.Tensor,
    plane_center: torch.Tensor,
    angle_thresh_deg: float = 10.0,
    dist_thresh: float = 0.02,
):
    """
    Select planar Gaussian indices by normal angle + distance from plane.

    Args:
        gaussians: GaussianModel (needs .normal and .get_xyz)
        plane_normal: (3,) world-space unit normal
        plane_center: (3,) world-space point on the plane

    Returns:
        planar_indices: (N,) integer tensor
    """
    device = gaussians.get_xyz.device
    plane_normal = F.normalize(plane_normal.to(device=device, dtype=torch.float32), dim=0)
    plane_center = plane_center.to(device=device, dtype=torch.float32)

    g_normals = gaussians.normal  # (G, 3), 2DGS: R[:, :, 2]

    cos_angle = torch.abs((g_normals * plane_normal[None, :]).sum(dim=-1))
    ang_thresh = math.cos(angle_thresh_deg / 180.0 * math.pi)
    mask_angle = cos_angle >= ang_thresh

    xyz = gaussians.get_xyz
    dist = torch.abs(((xyz - plane_center[None, :]) * plane_normal[None, :]).sum(dim=-1))
    mask_dist = dist <= dist_thresh

    return torch.nonzero(mask_angle & mask_dist, as_tuple=False).view(-1)


def merge_planes(plane1, plane2, normal_threshold=0.95, dist_threshold=0.1):
    """두 plane dict이 동일 평면인지 판단 (cosine similarity + distance)."""
    n1 = F.normalize(plane1["normal"], dim=0)
    n2 = F.normalize(plane2["normal"], dim=0)
    normal_sim = torch.abs(torch.dot(n1, n2)).item()
    dist = torch.abs(torch.dot(n1, plane1["center"] - plane2["center"])).item()
    return normal_sim > normal_threshold and dist < dist_threshold


def backproject_pixel_to_world(pixel_xy, depth, camera):
    """Pixel (x, y) + scalar depth → world-space 3D point."""
    device = depth.device if isinstance(depth, torch.Tensor) else torch.device("cuda")
    focal_x = camera.image_width  / (2 * torch.tan(torch.tensor(camera.FoVx / 2, device=device)))
    focal_y = camera.image_height / (2 * torch.tan(torch.tensor(camera.FoVy / 2, device=device)))

    x_cam = (pixel_xy[0] - camera.image_width  / 2) / focal_x * depth
    y_cam = (pixel_xy[1] - camera.image_height / 2) / focal_y * depth
    z_cam = depth

    point_cam = torch.stack([x_cam, y_cam, z_cam,
                              torch.ones_like(z_cam)], dim=0).float()
    world_view_inv = torch.inverse(camera.world_view_transform)
    return (world_view_inv @ point_cam)[:3]


def detect_planar_groups_from_normal(
    viewpoint_camera,
    gaussians,
    pipeline,
    background,
    render_func,
    num_groups: int = 2,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.60,
    angle_thresh_deg: float = 15.0,
    n_candidates: int = 12,
    alpha_thresh: float = 0.1,
):
    """
    2DGS-native planar detection using rendered rend_normal map.

    Unlike SAM-based detection (which starts from visual regions), this method
    starts directly from the Gaussian surfel normals to find geometrically planar
    areas, making it more aligned with 2DGS's explicit surface representation.

    Algorithm:
      1. Render → rend_normal (3,H,W) world-space + surf_depth (1,H,W)
      2. Quantize valid pixel normals into a 3D histogram (n_bins per axis)
      3. Find top-k dominant normal directions from histogram peaks
      4. For each dominant normal, build pixel mask by cosine similarity
      5. Fit plane center via depth backprojection of mask centroid

    Returns:
        List[Dict] with same schema as detect_planar_groups_from_depth_fast:
            'mask' (H,W bool), 'dominant_normal' (3,), 'mean_depth' (float), 'center' (3,)
    """
    with torch.no_grad():
        render_pkg = render_func(viewpoint_camera, gaussians, pipeline, background)

    normal_map = render_pkg.get("rend_normal", None)   # (3, H, W) world space
    depth_map  = render_pkg.get("surf_depth",  None)   # (1, H, W)
    alpha_map  = render_pkg.get("rend_alpha",  None)   # (1, H, W)

    if normal_map is None or depth_map is None:
        return []

    device = normal_map.device
    H, W = depth_map.shape[1], depth_map.shape[2]
    img_area = float(H * W)

    # Valid pixels: depth and alpha thresholds
    valid = depth_map[0] > 0.01
    if alpha_map is not None:
        valid = valid & (alpha_map[0] > alpha_thresh)
    if valid.sum() < 100:
        return []

    # Normalize normals and get valid subset (N, 3)
    n_map = F.normalize(normal_map, dim=0)          # (3, H, W)
    n_flat = n_map.permute(1, 2, 0)[valid]          # (N, 3)

    # Histogram quantization on the unit hemisphere.
    # Use absolute values so front/back of plane are treated the same.
    n_bins = 10
    bins = (n_flat.abs() * n_bins).long().clamp(0, n_bins - 1)   # (N, 3)
    bin_idx = bins[:, 0] * (n_bins * n_bins) + bins[:, 1] * n_bins + bins[:, 2]

    counts = torch.bincount(bin_idx, minlength=n_bins ** 3)
    top_bins = torch.topk(counts, k=min(n_candidates, int((counts > 0).sum()))).indices

    cos_thresh = math.cos(angle_thresh_deg / 180.0 * math.pi)
    valid_ys, valid_xs = torch.where(valid)

    planar_groups = []
    used_normals: list = []

    for b in top_bins:
        member = (bin_idx == b)
        if member.sum() < 10:
            continue

        dom_normal = F.normalize(n_flat[member].mean(0), dim=0)

        # Skip if too similar to an already-selected plane
        if any(torch.abs(torch.dot(dom_normal, prev)).item() > 0.95 for prev in used_normals):
            continue

        # Build full-resolution mask by cosine similarity on all valid pixels
        cos_sim = (n_flat * dom_normal.unsqueeze(0)).sum(dim=1).abs()  # (N,)
        in_plane = cos_sim >= cos_thresh

        area = int(in_plane.sum().item())
        area_ratio = area / img_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        mask_2d = torch.zeros(H, W, device=device, dtype=torch.bool)
        mask_2d[valid_ys[in_plane], valid_xs[in_plane]] = True

        # Plane center from backprojection of mask centroid + median depth
        cluster_depth = depth_map[0][mask_2d].median()
        ys_m, xs_m = torch.where(mask_2d)
        center_2d = torch.tensor(
            [xs_m.float().mean(), ys_m.float().mean()],
            dtype=torch.float32, device=device
        )
        center_3d = backproject_pixel_to_world(center_2d, cluster_depth, viewpoint_camera)

        # Ensure normal faces toward camera
        cam_dir = F.normalize(viewpoint_camera.camera_center - center_3d, dim=0)
        if (dom_normal * cam_dir).sum() < 0:
            dom_normal = -dom_normal

        used_normals.append(dom_normal)
        planar_groups.append({
            "mask":             mask_2d,
            "dominant_normal":  dom_normal,
            "mean_depth":       float(cluster_depth.item()),
            "center":           center_3d,
            "_area":            area,
        })

        if len(planar_groups) >= num_groups:
            break

    # Sort by area descending (largest planar region first)
    planar_groups.sort(key=lambda g: -g["_area"])
    for g in planar_groups:
        g.pop("_area", None)

    _planar_stats["total"] += 1
    if planar_groups:
        _planar_stats["detected"] += 1

    return planar_groups


def detect_planar_groups_from_depth_fast(
    viewpoint_camera,
    gaussians,
    pipeline,
    background,
    render_func,
    num_groups=1,
    min_group_size=1000,
    # SAM options
    use_sam: bool = False,
    sam_generator=None,
    sam_max_masks: int = 80,
    sam_min_area_ratio: float = 0.01,
    sam_max_area_ratio: float = 0.50,
    sam_plane_dot_thresh: float = 0.85,
    # selection options
    prefer_center: bool = False,
    prefer_near: bool = False,
):
    """
    SAM-based planar group detection using 2DGS surf_depth + rend_normal.

    Uses SVD plane fitting within each SAM mask for planarity scoring.
    Reads 'surf_depth' and 'rend_normal' from render_func output (2DGS keys).

    Returns:
        List[Dict]: each dict has keys:
            'mask' (H,W bool), 'dominant_normal' (3,), 'mean_depth' (float), 'center' (3,)
    """
    if not use_sam:
        return detect_planar_groups_from_normal(
            viewpoint_camera, gaussians, pipeline, background, render_func,
            num_groups=num_groups,
            min_area_ratio=0.02,
            max_area_ratio=0.60,
            angle_thresh_deg=15.0,
            n_candidates=12,
        )
    if sam_generator is None:
        raise ValueError("use_sam=True but sam_generator is None")

    with torch.no_grad():
        render_pkg = render_func(viewpoint_camera, gaussians, pipeline, background)

        # 2DGS uses 'surf_depth' and 'rend_normal'
        depth_map = render_pkg.get("surf_depth", None)
        normal_map = render_pkg.get("rend_normal", None)

        if depth_map is None or normal_map is None:
            return []

        if depth_map.dim() == 3:
            depth_map = depth_map[0]  # (H, W)

        H, W = depth_map.shape
        device = depth_map.device
        valid_depth = depth_map > 0.01

        img_t = getattr(viewpoint_camera, "original_image", None)
        if img_t is None:
            img_t = render_pkg.get("render", None)
        if img_t is None:
            return []

        img_np = (
            img_t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
        ).astype(np.uint8)

        torch.cuda.empty_cache()
        sam_masks = sam_generator.generate(img_np)
        sam_masks = sorted(sam_masks, key=lambda x: x.get("area", 0), reverse=True)[:sam_max_masks]

        img_area = float(H * W)
        screen_cx = (W - 1) * 0.5
        screen_cy = (H - 1) * 0.5
        planar_groups = []

        for m in sam_masks:
            mask_np = m.get("segmentation", None)
            if mask_np is None or mask_np.shape != (H, W):
                continue

            mask_2d = torch.from_numpy(mask_np).to(device=device).bool()
            mask_valid = mask_2d & valid_depth

            area = int(mask_valid.sum().item())
            if area < min_group_size:
                continue

            area_ratio = area / img_area
            if area_ratio < sam_min_area_ratio or area_ratio > sam_max_area_ratio:
                continue

            # SVD plane fit on backprojected points (camera space)
            ys, xs = torch.where(mask_valid)
            if ys.numel() < 10:
                continue

            depth_vals = depth_map[ys, xs]
            focal_x = viewpoint_camera.image_width  / (2 * torch.tan(torch.tensor(viewpoint_camera.FoVx / 2, device=device)))
            focal_y = viewpoint_camera.image_height / (2 * torch.tan(torch.tensor(viewpoint_camera.FoVy / 2, device=device)))

            x_cam = (xs.float() - viewpoint_camera.image_width  / 2) / focal_x * depth_vals
            y_cam = (ys.float() - viewpoint_camera.image_height / 2) / focal_y * depth_vals
            pts = torch.stack([x_cam, y_cam, depth_vals], dim=-1)  # (N, 3)

            centroid = pts.mean(dim=0, keepdim=True)
            _, _, Vh = torch.linalg.svd(pts - centroid, full_matrices=False)
            plane_n = Vh[-1]

            residuals = ((pts - centroid) * plane_n.unsqueeze(0)).sum(dim=-1).abs()
            mean_residual = residuals.mean().item()
            if mean_residual > 0.05:
                continue

            dot_med = 1.0 - mean_residual

            # Camera space → world space
            R_cam_to_world = viewpoint_camera.world_view_transform[:3, :3].T
            plane_n_world = F.normalize(R_cam_to_world @ plane_n, dim=0)
            cam_dir = F.normalize(viewpoint_camera.camera_center - centroid.squeeze(0), dim=0)
            if (plane_n_world * cam_dir).sum() < 0:
                plane_n_world = -plane_n_world

            cluster_depth = depth_map[mask_valid].median()
            ys, xs = torch.where(mask_valid)
            center_2d = torch.tensor(
                [xs.float().mean(), ys.float().mean()],
                dtype=torch.float32, device=device
            )
            dx = (center_2d[0] - screen_cx) / float(W)
            dy = (center_2d[1] - screen_cy) / float(H)
            center_dist = float(torch.sqrt(dx * dx + dy * dy).item())
            center_3d = backproject_pixel_to_world(center_2d, cluster_depth, viewpoint_camera)

            planar_groups.append({
                "mask": mask_2d,
                "dominant_normal": plane_n_world,
                "mean_depth": float(cluster_depth.item()),
                "center": center_3d,
                "_score": dot_med,
                "_area": area,
                "_center_dist": center_dist,
            })

        if len(planar_groups) == 0:
            _planar_stats["total"] += 1
            return []

        if prefer_center and prefer_near:
            planar_groups.sort(key=lambda g: (-g["_score"], g["_center_dist"], -g["_area"], g["mean_depth"]))
        elif prefer_center:
            planar_groups.sort(key=lambda g: (-g["_score"], g["_center_dist"], -g["_area"]))
        elif prefer_near:
            planar_groups.sort(key=lambda g: (-g["_score"], -g["_area"], g["mean_depth"]))
        else:
            planar_groups.sort(key=lambda g: (-g["_score"], -g["_area"]))

        planar_groups = planar_groups[:num_groups]

        _planar_stats["total"] += 1
        _planar_stats["detected"] += 1

        for g in planar_groups:
            g.pop("_score", None)
            g.pop("_area", None)
            g.pop("_center_dist", None)

        return planar_groups
