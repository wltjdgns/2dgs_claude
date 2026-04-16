"""
[Manual] 단일 뷰 bbox → 3D 평면 추출 → 전체 카메라 planar_cache 생성

사용법:
  1. 렌더된 이미지 (train/ours_30000/renders/00030.png) 로컬로 받아서 열기
  2. 태블릿 픽셀 좌표 확인 (x1 y1 x2 y2)
  3. 실행:
     python scripts/manual_planar_cache.py \\
         -m ./output/1_custom/0414_r --iteration 30000 \\
         --cam_idx 30 --bbox 400 200 650 500

  4. render.py에 캐시 주입:
     python render.py -m ./output/1_custom/0414_r --enable_2pass \\
         --planar_cache_path ./output/1_custom/0414_r/planar_cache_manual.pt

bbox 좌표: --bbox x1 y1 x2 y2  (left top right bottom, 픽셀 단위)
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from utils.planar_utils import backproject_pixel_to_world


# ── 전체 픽셀 world-space 좌표 복원 ──────────────────────────────────────────
def reconstruct_xyz_map(cam, surf_depth):
    """surf_depth (1,H,W) → world-space xyz (3,H,W)"""
    device = surf_depth.device
    depth  = surf_depth[0]           # (H, W)
    H, W   = depth.shape

    fx = cam.image_width  / (2 * math.tan(cam.FoVx / 2))
    fy = cam.image_height / (2 * math.tan(cam.FoVy / 2))

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x_cam = (grid_x - W / 2) / fx * depth
    y_cam = (grid_y - H / 2) / fy * depth
    ones  = torch.ones_like(depth)

    pts_cam   = torch.stack([x_cam, y_cam, depth, ones], dim=0)  # (4,H,W)
    c2w       = torch.inverse(cam.world_view_transform.T)          # C2W = (W2C.T).T.inv = W2C.inv
    pts_world = (c2w @ pts_cam.reshape(4, -1)).reshape(4, H, W)
    return pts_world[:3] * (depth > 0.01).unsqueeze(0).float()   # (3,H,W)


# ── 폴리곤 안 + 평면 위 가우시안 인덱스 추출 ──────────────────────────────────
def extract_gaussians_in_polygon(gaussians, ref_cam, region_mask,
                                  plane_normal, plane_center, dist_thresh):
    """
    기준 카메라 시점에서 (a) 폴리곤 영역에 투영되고
    (b) 평면까지 거리 < dist_thresh 인 가우시안 인덱스 반환.

    Args:
        gaussians:     GaussianModel
        ref_cam:       기준 카메라
        region_mask:   (H, W) bool — 폴리곤 영역
        plane_normal:  (3,) world
        plane_center:  (3,) world
        dist_thresh:   float — 평면까지 거리 한계 (world unit)

    Returns:
        indices: (K,) long — 선택된 가우시안 인덱스
    """
    xyz    = gaussians.get_xyz                       # (N, 3) world
    device = xyz.device
    N      = xyz.shape[0]
    H, W   = region_mask.shape

    # (a) 평면까지 거리 필터
    diff       = xyz - plane_center.to(device).view(1, 3)
    dist       = (diff * plane_normal.to(device).view(1, 3)).sum(-1).abs()  # (N,)
    near_plane = dist < dist_thresh

    # (b) world → camera: W2C @ pts_col = world_view_transform.T @ pts_col
    ones        = torch.ones(N, 1, device=device)
    pts_world_h = torch.cat([xyz, ones], dim=-1).T        # (4, N)
    pts_cam_h   = ref_cam.world_view_transform.T @ pts_world_h  # (4, N)
    pts_cam     = pts_cam_h[:3, :].T                       # (N, 3)

    in_front = pts_cam[:, 2] > 0.01

    # camera → image (pinhole, same convention as reconstruct_xyz_map)
    fx = ref_cam.image_width  / (2 * math.tan(ref_cam.FoVx / 2))
    fy = ref_cam.image_height / (2 * math.tan(ref_cam.FoVy / 2))

    z_safe = pts_cam[:, 2].clamp(min=0.01)
    u      = pts_cam[:, 0] / z_safe * fx + W / 2
    v      = pts_cam[:, 1] / z_safe * fy + H / 2

    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    u_int = u.round().long().clamp(0, W - 1)
    v_int = v.round().long().clamp(0, H - 1)

    in_polygon = torch.zeros(N, dtype=torch.bool, device=device)
    valid_idx  = in_front & in_image
    in_polygon[valid_idx] = region_mask[v_int[valid_idx], u_int[valid_idx]]

    selected = near_plane & in_front & in_polygon
    print(f"  가우시안 필터링: total={N}  near_plane={int(near_plane.sum())}  "
          f"in_polygon={int(in_polygon.sum())}  selected={int(selected.sum())}")
    return torch.where(selected)[0]


# ── 3D 평면을 카메라 뷰에 투영 → 마스크 생성 (v1 레거시) ─────────────────────
def project_plane_to_mask(cam, pkg, plane_normal, plane_center,
                           dist_thresh: float, normal_deg: float):
    """
    3D 평면(plane_normal, plane_center)을 cam 시점에 투영하여 (H,W) bool mask 반환.

    조건:
      1. surf_depth 기반 world-space 거리 < dist_thresh
      2. rend_normal과 plane_normal의 각도 < normal_deg
    """
    depth_map  = pkg.get("surf_depth")
    normal_map = pkg.get("rend_normal")
    if depth_map is None:
        return None

    device = depth_map.device
    H, W   = depth_map.shape[1], depth_map.shape[2]
    valid  = depth_map[0] > 0.01                         # (H,W)

    xyz    = reconstruct_xyz_map(cam, depth_map)          # (3,H,W)
    diff   = xyz - plane_center.view(3, 1, 1)
    dist   = (diff * plane_normal.view(3, 1, 1)).sum(0).abs()  # (H,W)
    dist_mask = dist < dist_thresh

    if normal_map is not None:
        n_map      = F.normalize(normal_map, dim=0)
        cos_thresh = math.cos(math.radians(normal_deg))
        normal_mask = (n_map * plane_normal.view(3, 1, 1)).sum(0).abs() >= cos_thresh
    else:
        normal_mask = torch.ones(H, W, device=device, dtype=torch.bool)

    return valid & dist_mask & normal_mask


# ── 기준 뷰 bbox에서 3D 평면 추출 ────────────────────────────────────────────
def extract_plane_from_bbox(cam, pkg, bbox, save_debug_img=None):
    """
    bbox (x1,y1,x2,y2) 픽셀 영역에서 3D 평면(normal, center)을 추출.

    normal: bbox 내 rend_normal 평균 (normal-based가 depth SVD보다 안정적)
    center: bbox centroid 픽셀의 median depth를 역투영
    """
    depth_map  = pkg.get("surf_depth")
    normal_map = pkg.get("rend_normal")
    if depth_map is None or normal_map is None:
        raise RuntimeError("surf_depth 또는 rend_normal이 render_pkg에 없습니다.")

    from PIL import Image, ImageDraw

    device = depth_map.device
    H, W   = depth_map.shape[1], depth_map.shape[2]
    valid  = depth_map[0] > 0.01                         # (H,W)

    # ── 폴리곤 마스크 생성 (PIL) ──────────────────────────────────────────
    # bbox: [x1,y1,x2,y2] → 직사각형  |  poly: [x1,y1,...,x4,y4] → 임의 사각형
    poly_img  = Image.new("L", (W, H), 0)
    draw      = ImageDraw.Draw(poly_img)
    draw.polygon(bbox, fill=255)                          # bbox는 꼭짓점 리스트
    region_mask = torch.from_numpy(
        np.array(poly_img) > 0
    ).to(device)                                          # (H,W) bool

    region_valid = region_mask & valid
    n_pixels     = int(region_valid.sum().item())
    n_region     = int(region_mask.sum().item())
    print(f"  지정 영역 유효 픽셀: {n_pixels} / {n_region}")
    if n_pixels < 10:
        raise RuntimeError("지정 영역 내 유효 픽셀이 너무 적습니다. 좌표를 확인하세요.")

    # normal: 최빈값 (구면 좌표 bin → 가장 많은 bin의 평균)
    n_map       = F.normalize(normal_map, dim=0)          # (3,H,W)
    n_in_region = n_map[:, region_valid]                  # (3,N)

    BIN_DEG = 5                                           # 5° 단위로 양자화
    nz       = n_in_region[2].clamp(-1, 1)
    nx, ny   = n_in_region[0], n_in_region[1]
    theta_bin = (torch.acos(nz)                    * 180 / math.pi / BIN_DEG).long()
    phi_bin   = ((torch.atan2(ny, nx) * 180 / math.pi + 360) % 360 / BIN_DEG).long()
    bin_idx   = theta_bin * 1000 + phi_bin          # 유일한 bin ID

    counts    = torch.bincount(bin_idx.clamp(min=0))
    mode_bin  = counts.argmax().item()
    mode_mask = (bin_idx == mode_bin)
    plane_normal = F.normalize(n_in_region[:, mode_mask].mean(1), dim=0)
    print(f"  mode bin 픽셀: {int(mode_mask.sum())} / {n_in_region.shape[1]}"
          f"  ({int(mode_mask.sum()) / n_in_region.shape[1] * 100:.1f}%)")

    # center: 영역 centroid 역투영 (cam_dir 계산보다 먼저)
    ys_b, xs_b    = torch.where(region_valid)
    cluster_depth = depth_map[0][region_valid].median()
    center_2d     = torch.tensor(
        [xs_b.float().mean(), ys_b.float().mean()],
        dtype=torch.float32, device=device,
    )
    plane_center = backproject_pixel_to_world(center_2d, cluster_depth, cam)

    # normal이 카메라를 향하도록 (plane_center 기준)
    cam_dir = F.normalize(cam.camera_center - plane_center, dim=0)
    if (plane_normal * cam_dir).sum() < 0:
        plane_normal = -plane_normal

    # 법선 일관성 확인
    cos_vals = (n_in_region.T * plane_normal).sum(-1)
    ang_std  = torch.acos(cos_vals.clamp(-1, 1)).std().item() * 180 / math.pi
    print(f"  normal consistency (std): {ang_std:.1f}°  "
          f"{'✓ 평면적' if ang_std < 15 else '△ 노이즈 있음'}")
    print(f"  plane_normal: {plane_normal.cpu().numpy().round(3)}")
    print(f"  plane_center: {plane_center.cpu().numpy().round(3)}")

    # 디버그 이미지 저장 (폴리곤 + normal 방향 + center 시각화)
    if save_debug_img is not None:
        img = pkg["render"].detach().permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8).copy()
        overlay      = Image.fromarray(img)
        draw_overlay = ImageDraw.Draw(overlay)

        # 폴리곤 테두리 (빨간색)
        draw_overlay.polygon(bbox, outline=(255, 0, 0))

        # plane_center를 이미지에 투영 → 파란 점
        device = depth_map.device
        cx = torch.tensor([xs_b.float().mean().item()], device=device)
        cy = torch.tensor([ys_b.float().mean().item()], device=device)
        cx_i, cy_i = int(cx.item()), int(cy.item())
        r = 6
        draw_overlay.ellipse([cx_i - r, cy_i - r, cx_i + r, cy_i + r], fill=(0, 0, 255))

        # world_view_transform = W2C.T → [:3,:3] = C2W_R → .T = W2C_R
        W2C_R = cam.world_view_transform[:3, :3].T     # (3,3) world→cam rotation

        def _proj_arrow(d_world, color, label, start_xy, arrow_len=70, arriving=False):
            """world 방향 벡터를 카메라 접선면에 투영해 화살표 그리기"""
            d_cam = W2C_R @ d_world.to(device)         # (3,) camera space
            dx = int(d_cam[0].item() * arrow_len)
            dy = int(d_cam[1].item() * arrow_len)      # 이미지 y = cam y (동일 방향)
            sx, sy = start_xy
            if arriving:                                # 화살표가 start_xy로 도달
                ex, ey = sx, sy
                sx, sy = sx - dx, sy - dy
            else:                                       # 화살표가 start_xy에서 출발
                ex, ey = sx + dx, sy + dy
            draw_overlay.line([sx, sy, ex, ey], fill=color, width=3)
            draw_overlay.ellipse([ex-4, ey-4, ex+4, ey+4], fill=color)

        arrow_len = 70
        # 초록: plane_normal
        n_world = plane_normal.to(device)
        _proj_arrow(n_world, (0,255,0), "normal", (cx_i, cy_i), arrow_len)

        # 노랑: input 방향 (카메라 → center, arriving 화살표)
        d_in  = F.normalize(plane_center - cam.camera_center, dim=0)
        _proj_arrow(d_in, (255,255,0), "input", (cx_i, cy_i), arrow_len, arriving=True)

        # 청록: output/반사 방향 (center에서 d_refl 방향으로 출발)
        d_refl = F.normalize(d_in - 2 * torch.dot(d_in, plane_normal.to(device)) * plane_normal.to(device), dim=0)
        _proj_arrow(d_refl, (0,255,255), "refl", (cx_i, cy_i), arrow_len)

        overlay.save(save_debug_img)
        print(f"  디버그 이미지 저장: {save_debug_img}")
        print(f"    (빨강=폴리곤  파랑●=center  초록→=normal  노랑→=input  청록→=반사방향)")

    return plane_normal, plane_center


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = ArgumentParser(description="Manual planar cache generator")
    model    = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration",  default=-1,   type=int)
    parser.add_argument("--cam_idx",    required=True, type=int,
                        help="기준 카메라 index (train split 기준)")
    parser.add_argument("--bbox", nargs=4, type=int,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="직사각형 영역 (left top right bottom)")
    parser.add_argument("--poly", nargs=8, type=int,
                        metavar=("X1","Y1","X2","Y2","X3","Y3","X4","Y4"),
                        help="비정형 사각형 4꼭짓점 (좌상→우상→우하→좌하 순, 8개 값)")
    parser.add_argument("--dist_thresh", default=0.05, type=float,
                        help="평면 거리 허용 오차 (world unit). 너무 크면 과탐지.")
    parser.add_argument("--normal_deg",  default=20.0, type=float,
                        help="normal 각도 허용 오차 (degree)")
    parser.add_argument("--split",      default="train",
                        choices=["train", "test", "both"])
    parser.add_argument("--out_name",   default="planar_cache_manual.pt", type=str)
    args = get_combined_args(parser)

    safe_state(False)

    dataset   = model.extract(args)
    pipe      = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene     = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg        = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    test_cams  = scene.getTestCameras()

    cam_idx    = getattr(args, 'cam_idx',    0)

    # --poly 우선, 없으면 --bbox → PIL polygon 꼭짓점 리스트로 통일
    raw_poly = getattr(args, 'poly', None)
    raw_bbox = getattr(args, 'bbox', None)
    if raw_poly is not None:
        # 8개 값 → [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        xs = raw_poly[0::2]
        ys = raw_poly[1::2]
        bbox = list(zip(xs, ys))
    elif raw_bbox is not None:
        x1, y1, x2, y2 = raw_bbox
        bbox = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]   # 직사각형
    else:
        raise ValueError("--bbox 또는 --poly 중 하나는 반드시 지정해야 합니다.")
    dist_thresh = getattr(args, 'dist_thresh', 0.05)
    normal_deg  = getattr(args, 'normal_deg',  20.0)
    split       = getattr(args, 'split',      'train')
    out_name    = getattr(args, 'out_name',   'planar_cache_manual.pt')

    # ── Step 1: 기준 카메라에서 3D 평면 추출 ──────────────────────────────
    ref_cam = train_cams[cam_idx]
    print(f"\n[manual_planar_cache] 기준 카메라: [{cam_idx}] {ref_cam.image_name}")
    print(f"  bbox: x1={bbox[0]} y1={bbox[1]} x2={bbox[2]} y2={bbox[3]}")

    with torch.no_grad():
        ref_pkg = render(ref_cam, gaussians, pipe, bg)

    debug_path = os.path.join(args.model_path, "planar_cache_manual_debug.png")
    plane_normal, plane_center = extract_plane_from_bbox(
        ref_cam, ref_pkg, bbox, save_debug_img=debug_path
    )

    # ── Step 2: 폴리곤 꼭짓점 4개를 3D로 역투영 ──────────────────────────────
    depth_map = ref_pkg["surf_depth"][0]           # (H, W)
    H, W = depth_map.shape
    polygon_corners_3d = []
    cluster_depth = depth_map[depth_map > 0.01].median()

    for (u, v) in bbox:
        u_c = min(max(int(round(u)), 0), W - 1)
        v_c = min(max(int(round(v)), 0), H - 1)
        d = depth_map[v_c, u_c]
        if d < 0.01:
            d = cluster_depth           # 유효 depth 없으면 평균 depth 사용
        corner_3d = backproject_pixel_to_world(
            torch.tensor([float(u), float(v)], device="cuda"), d.cuda(), ref_cam
        )
        polygon_corners_3d.append(corner_3d.cpu())

    corners_tensor = torch.stack(polygon_corners_3d)   # (4, 3)
    print(f"  3D 폴리곤 꼭짓점:\n{corners_tensor.numpy().round(3)}")

    # ── Step 3: v3 포맷으로 저장 ───────────────────────────────────────────
    cache = {
        "version":            3,
        "polygon_corners_3d": corners_tensor,
        "dominant_normal":    plane_normal.cpu(),
        "center":             plane_center.cpu(),
    }
    out_path = os.path.join(args.model_path, out_name)
    torch.save(cache, out_path)

    print(f"\n[manual_planar_cache] 완료 (v3: polygon-projection)")
    print(f"  저장 → {out_path}")
    print(f"  디버그 이미지 → {debug_path}")
    print(f"\n  다음 단계:")
    print(f"    python render.py -m {args.model_path} --enable_2pass --planar_cache_path {out_path}")
