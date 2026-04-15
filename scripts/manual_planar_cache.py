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
    c2w       = torch.inverse(cam.world_view_transform)           # (4,4)
    pts_world = (c2w @ pts_cam.reshape(4, -1)).reshape(4, H, W)
    return pts_world[:3] * (depth > 0.01).unsqueeze(0).float()   # (3,H,W)


# ── 3D 평면을 카메라 뷰에 투영 → 마스크 생성 ─────────────────────────────────
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

    device    = depth_map.device
    H, W      = depth_map.shape[1], depth_map.shape[2]
    x1, y1, x2, y2 = bbox

    # 경계 클램핑
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)

    valid = depth_map[0] > 0.01                          # (H,W)

    bbox_mask = torch.zeros(H, W, device=device, dtype=torch.bool)
    bbox_mask[y1:y2, x1:x2] = True
    bbox_valid = bbox_mask & valid

    n_pixels = int(bbox_valid.sum().item())
    print(f"  bbox 유효 픽셀: {n_pixels} / {(y2-y1)*(x2-x1)}")
    if n_pixels < 10:
        raise RuntimeError("bbox 내 유효 픽셀이 너무 적습니다. bbox 좌표를 확인하세요.")

    # normal: bbox 평균
    n_map       = F.normalize(normal_map, dim=0)          # (3,H,W)
    n_in_bbox   = n_map[:, bbox_valid]                    # (3,N)
    plane_normal = F.normalize(n_in_bbox.mean(1), dim=0)  # (3,)

    # normal이 카메라를 향하도록
    cam_dir = F.normalize(cam.camera_center - plane_normal, dim=0)
    if (plane_normal * cam_dir).sum() < 0:
        plane_normal = -plane_normal

    # center: bbox centroid 역투영
    ys_b, xs_b   = torch.where(bbox_valid)
    cluster_depth = depth_map[0][bbox_valid].median()
    center_2d     = torch.tensor(
        [xs_b.float().mean(), ys_b.float().mean()],
        dtype=torch.float32, device=device,
    )
    plane_center = backproject_pixel_to_world(center_2d, cluster_depth, cam)

    # 법선 일관성 확인 (bbox 내 angular std)
    cos_vals  = (n_in_bbox.T * plane_normal).sum(-1)       # (N,)
    ang_std   = torch.acos(cos_vals.clamp(-1, 1)).std().item() * 180 / math.pi
    print(f"  normal consistency (std): {ang_std:.1f}°  "
          f"{'✓ 평면적' if ang_std < 15 else '△ 노이즈 있음'}")
    print(f"  plane_normal: {plane_normal.cpu().numpy().round(3)}")
    print(f"  plane_center: {plane_center.cpu().numpy().round(3)}")

    # 디버그 이미지 저장 (bbox overlay)
    if save_debug_img is not None:
        from PIL import Image
        img = pkg["render"].detach().permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8).copy()
        # bbox 테두리 그리기 (빨간색)
        img[y1:y2, x1:x1+3]   = [255, 0, 0]
        img[y1:y2, x2-3:x2]   = [255, 0, 0]
        img[y1:y1+3, x1:x2]   = [255, 0, 0]
        img[y2-3:y2, x1:x2]   = [255, 0, 0]
        Image.fromarray(img).save(save_debug_img)
        print(f"  디버그 이미지 저장: {save_debug_img}")

    return plane_normal, plane_center


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = ArgumentParser(description="Manual planar cache generator")
    model    = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration",  default=-1,   type=int)
    parser.add_argument("--cam_idx",    required=True, type=int,
                        help="기준 카메라 index (train split 기준)")
    parser.add_argument("--bbox",       required=True, nargs=4, type=int,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="태블릿 픽셀 bbox (left top right bottom)")
    parser.add_argument("--dist_thresh", default=0.05, type=float,
                        help="평면 거리 허용 오차 (world unit). 너무 크면 과탐지.")
    parser.add_argument("--normal_deg",  default=20.0, type=float,
                        help="normal 각도 허용 오차 (degree)")
    parser.add_argument("--split",      default="train",
                        choices=["train", "test", "both"])
    parser.add_argument("--out_name",   default="planar_cache_manual.pt", type=str)
    args = get_combined_args(parser)

    safe_state(True)

    dataset   = model.extract(args)
    pipe      = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene     = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg        = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    test_cams  = scene.getTestCameras()

    bbox       = getattr(args, 'bbox',       None)
    cam_idx    = getattr(args, 'cam_idx',    0)
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

    # ── Step 2: 전체 카메라에 평면 투영 → 마스크 생성 ─────────────────────
    splits = []
    if split in ("train", "both"):
        splits.append(("train", train_cams))
    if split in ("test", "both"):
        splits.append(("test",  test_cams))

    cache: dict = {}
    n_detected  = 0

    for split_name, cameras in splits:
        print(f"\n[manual_planar_cache] {split_name} split ({len(cameras)} cameras) "
              f"dist_thresh={dist_thresh}  normal_deg={normal_deg}°")

        for idx, cam in enumerate(tqdm(cameras, desc=split_name)):
            cam_name = getattr(cam, "image_name", f"cam_{idx}")

            with torch.no_grad():
                pkg = render(cam, gaussians, pipe, bg)

            mask = project_plane_to_mask(
                cam, pkg, plane_normal, plane_center,
                dist_thresh=dist_thresh,
                normal_deg=normal_deg,
            )

            if mask is not None and mask.sum() > 100:
                depth_map    = pkg["surf_depth"]
                cluster_depth = float(depth_map[0][mask].median().item())
                ys_m, xs_m   = torch.where(mask)
                center_2d    = torch.tensor(
                    [xs_m.float().mean(), ys_m.float().mean()],
                    dtype=torch.float32, device=mask.device,
                )
                center_3d = backproject_pixel_to_world(
                    center_2d, torch.tensor(cluster_depth, device=mask.device), cam
                )
                cache[cam_name] = [{
                    "mask":            mask.cpu(),
                    "dominant_normal": plane_normal.cpu(),
                    "mean_depth":      cluster_depth,
                    "center":          center_3d.cpu(),
                }]
                n_detected += 1
            else:
                cache[cam_name] = []   # 이 뷰에서는 평면 안 보임

    # ── Step 3: 저장 ───────────────────────────────────────────────────────
    out_path = os.path.join(args.model_path, out_name)
    torch.save(cache, out_path)

    print(f"\n[manual_planar_cache] 완료")
    print(f"  평면 탐지된 카메라: {n_detected} / {len(cache)}")
    print(f"  저장 → {out_path}")
    print(f"  디버그 이미지 → {debug_path}")
    print(f"\n  다음 단계:")
    print(f"    python render.py -m {args.model_path} --enable_2pass "
          f"--planar_cache_path {out_path}")
