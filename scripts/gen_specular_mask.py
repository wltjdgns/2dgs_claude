"""
[C] Specular-guided Planar Mask Pre-computation Script

자동으로 spec_thresh를 계산(Otsu's method)하고, 전체 카메라에 대해
specularity 기반 planar mask를 생성하여 planar_cache_specular.pt로 저장.

Run BEFORE render.py:
    python scripts/gen_specular_mask.py -m <model_path> --iteration 30000

Then use in render.py:
    python render.py -m <model_path> --enable_2pass \\
        --planar_cache_path <model_path>/planar_cache_specular.pt

--spec_thresh를 명시하면 Otsu 계산을 건너뜁니다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from utils.planar_utils import detect_planar_from_specularity


def otsu_threshold(values: np.ndarray) -> float:
    """
    Otsu's method: finds threshold that maximizes between-class variance.
    Input: 1-D float array in [0, 1].
    """
    hist, bin_edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    best_thresh = 0.0
    best_var = 0.0
    w0, sum0 = 0.0, 0.0
    total_mean = (hist * bin_centers).sum() / total

    for i in range(len(hist)):
        w0 += hist[i] / total
        if w0 == 0:
            continue
        w1 = 1.0 - w0
        if w1 == 0:
            break
        sum0 += hist[i] * bin_centers[i] / total
        mu0 = sum0 / w0
        mu1 = (total_mean - sum0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_thresh = bin_centers[i]

    return float(best_thresh)


def compute_auto_thresh(cameras, gaussians, pipe, bg, sample_n: int = 10) -> float:
    """Sample frames and compute Otsu threshold on specularity distribution."""
    step = max(1, len(cameras) // sample_n)
    sampled = cameras[::step][:sample_n]
    all_spec = []

    print(f"[auto-thresh] Sampling {len(sampled)} frames for Otsu threshold...")
    for cam in tqdm(sampled, desc="sampling"):
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, bg)
        r = pkg.get("roughnessmap")
        if r is None:
            continue
        spec = (1.0 - r.clamp(0, 1)).flatten().cpu().numpy()
        all_spec.append(spec)

    if not all_spec:
        print("[auto-thresh] WARNING: roughnessmap missing — fallback to 0.5")
        return 0.5

    all_spec = np.concatenate(all_spec)
    thresh = otsu_threshold(all_spec)

    # Diagnostics
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pct = 100.0 * (all_spec > t).mean()
        print(f"  spec > {t:.1f} : {pct:.1f}% of pixels")
    print(f"  mean specularity : {all_spec.mean():.3f}")
    print(f"  Otsu threshold   : {thresh:.3f}  "
          f"({100*(all_spec > thresh).mean():.1f}% of pixels above)")

    return thresh


if __name__ == "__main__":
    parser = ArgumentParser(description="Pre-compute specular planar mask cache")
    model    = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration",   default=-1,    type=int)
    parser.add_argument("--split",       default="train", choices=["train", "test", "both"])
    parser.add_argument("--spec_thresh", default=None,  type=float,
                        help="Specularity threshold. If omitted, auto-computed via Otsu.")
    parser.add_argument("--sample_n",   default=10,    type=int,
                        help="Frames to sample for Otsu threshold estimation")
    parser.add_argument("--normal_deg",  default=15.0,  type=float)
    parser.add_argument("--min_area",    default=0.005, type=float)
    parser.add_argument("--max_area",    default=0.5,   type=float)
    parser.add_argument("--num_groups",  default=3,     type=int)
    parser.add_argument("--out_name",    default="planar_cache_specular.pt", type=str)
    args = get_combined_args(parser)

    safe_state(True)

    dataset   = model.extract(args)
    pipe      = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene     = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg        = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    test_cams  = scene.getTestCameras()

    # ── Auto-compute spec_thresh if not given ──────────────────────────────
    spec_thresh_arg = getattr(args, 'spec_thresh', None)
    if spec_thresh_arg is None:
        print(f"\n{'='*60}")
        print(f"  Auto spec_thresh  (model: {args.model_path})")
        print(f"{'='*60}")
        sample_n = getattr(args, 'sample_n', 10)
        spec_thresh = compute_auto_thresh(train_cams, gaussians, pipe, bg, sample_n)
        print(f"{'='*60}\n")
    else:
        spec_thresh = spec_thresh_arg
        print(f"[gen_specular_mask] Using manual spec_thresh={spec_thresh}")

    split     = getattr(args, 'split',      'train')
    normal_deg = getattr(args, 'normal_deg', 15.0)
    min_area  = getattr(args, 'min_area',   0.005)
    max_area  = getattr(args, 'max_area',   0.5)
    num_groups = getattr(args, 'num_groups', 3)
    out_name  = getattr(args, 'out_name',   'planar_cache_specular.pt')

    # ── Generate masks ─────────────────────────────────────────────────────
    splits = []
    if split in ("train", "both"):
        splits.append(("train", train_cams))
    if split in ("test", "both"):
        splits.append(("test",  test_cams))

    cache: dict = {}

    for split_name, cameras in splits:
        print(f"[gen_specular_mask] {split_name} split ({len(cameras)} cameras) "
              f"spec_thresh={spec_thresh:.3f}")

        for idx, cam in enumerate(tqdm(cameras, desc=split_name)):
            cam_name = getattr(cam, "image_name", f"cam_{idx}")

            with torch.no_grad():
                pkg = render(cam, gaussians, pipe, bg)

            groups = detect_planar_from_specularity(
                cam, pkg,
                spec_thresh=spec_thresh,
                normal_consistency_deg=normal_deg,
                min_area_ratio=min_area,
                max_area_ratio=max_area,
                num_groups=num_groups,
            )

            cache[cam_name] = [
                {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                 for k, v in g.items()}
                for g in groups
            ]

    out_path = os.path.join(args.model_path, out_name)
    torch.save(cache, out_path)

    n_detected = sum(len(v) > 0 for v in cache.values())
    print(f"\n[gen_specular_mask] Done.")
    print(f"  spec_thresh used  : {spec_thresh:.3f}")
    print(f"  Cameras detected  : {n_detected} / {len(cache)}")
    print(f"  Saved → {out_path}")
    print(f"\n  Next:")
    print(f"    python render.py -m {args.model_path} --enable_2pass "
          f"--planar_cache_path {out_path}")
