#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor
from utils.render_utils import generate_path, create_videos, save_img_u8

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (alias for --model_path)")
    # PBR / 2-pass options
    parser.add_argument("--enable_2pass", action="store_true",
                        help="Enable 2-pass planar reflection rendering (loads planar_cache.pt from model dir)")
    parser.add_argument("--metalnet_ckpt", type=str, default="./pretrain/MetalicNet/metal.pth",
                        help="Path to MetalicNet checkpoint for metal map inference")
    parser.add_argument("--lambda_weight", type=float, default=0.5,
                        help="Specular blend weight in render_2pass BRDF composition")
    args = get_combined_args(parser)
    if args.output_dir:
        args.model_path = args.output_dir
    print("Rendering " + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir  = os.path.join(args.model_path, 'test',  "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

    # ── optional: MetalicNET ────────────────────────────────────────────
    metalnet = None
    if args.metalnet_ckpt:
        from utils.metalnet_utils import load_metalnet, predict_metal_map, metalprob_to_f0_rgb
        metalnet = load_metalnet(args.metalnet_ckpt)
        print(f"[MetalNet] loaded from {args.metalnet_ckpt}")

    # ── optional: SAM for 2-pass planar detection ───────────────────────
    sam_generator = None
    sam_ckpt = getattr(args, 'sam_ckpt', None)
    sam_model_type = getattr(args, 'sam_model_type', 'vit_h')
    enable_2pass = getattr(args, 'enable_2pass', False)
    if enable_2pass and sam_ckpt:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt).cuda()
        sam_generator = SamAutomaticMaskGenerator(sam)
        print(f"[SAM] loaded {sam_model_type} from {sam_ckpt}")

    # ── Load planar cache saved by train.py (preferred over re-detection) ──
    planar_cache = None
    if enable_2pass:
        cache_path = os.path.join(args.model_path, "planar_cache.pt")
        if os.path.exists(cache_path):
            raw_cache = torch.load(cache_path, map_location="cpu")
            planar_cache = {
                cam_name: [
                    {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in g.items()}
                    for g in groups
                ]
                for cam_name, groups in raw_cache.items()
            }
            print(f"[Planar] loaded cache ({len(planar_cache)} cameras) from {cache_path}")

    def render_func(cam, gs, pipe_, bg, scaling_modifier=1.0, override_color=None):
        return render(cam, gs, pipe_, bg, scaling_modifier=scaling_modifier, override_color=override_color)

    def _run_camera_set(cameras, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        gaussExtractor.reconstruction(cameras)
        gaussExtractor.export_image(out_dir)

        if not (enable_2pass or metalnet):
            return

        from utils.planar_utils import detect_planar_groups_from_depth_fast
        from gaussian_renderer.render_2pass import render_2pass
        twopass_path    = os.path.join(out_dir, "vis", "2pass")
        planar_path     = os.path.join(out_dir, "vis", "planar_mask")
        metal_path      = os.path.join(out_dir, "vis", "metal")
        f0_path         = os.path.join(out_dir, "vis", "f0")
        for p in [twopass_path, planar_path, metal_path, f0_path]:
            os.makedirs(p, exist_ok=True)

        for idx, cam in tqdm(enumerate(cameras), desc="PBR / 2-pass export"):
            stem = '{0:05d}'.format(idx)
            base_pkg = render(cam, gaussians, pipe, background)

            # ── MetalicNET (먼저 실행: F0/metal_map을 2pass에 주입하기 위해) ──
            metal_map = None
            if metalnet is not None:
                from utils.metalnet_utils import predict_metal_map, metalprob_to_f0_rgb
                metal_map = predict_metal_map(metalnet, base_pkg)
                if metal_map is not None:
                    save_img_u8(metal_map.detach().expand(3,-1,-1).permute(1,2,0).cpu().numpy(),
                                os.path.join(metal_path, f'{stem}.png'))
                    f0 = metalprob_to_f0_rgb(base_pkg, metal_map)
                    if f0 is not None:
                        save_img_u8(f0.detach().permute(1,2,0).cpu().numpy(),
                                    os.path.join(f0_path, f'{stem}.png'))
                        # f0_map을 base_pkg에 주입 → render_2pass에서 F0로 사용
                        base_pkg["f0_map"] = f0

            # ── Resolve planar groups: cache → SAM → normal-based fallback ──
            cam_name = getattr(cam, "image_name", f"cam_{idx}")
            if planar_cache is not None:
                planar_groups = planar_cache.get(cam_name, [])
            elif sam_generator is not None:
                planar_groups = detect_planar_groups_from_depth_fast(
                    cam, gaussians, pipe, background, render_func,
                    use_sam=True, sam_generator=sam_generator,
                )
            else:
                planar_groups = detect_planar_groups_from_depth_fast(
                    cam, gaussians, pipe, background, render_func,
                    use_sam=False,
                )

            # ── 2-pass rendering (metal_map + f0_map 주입된 base_pkg 사용) ──
            if enable_2pass:
                pkg = render_2pass(
                    cam, gaussians, pipe, background,
                    render_func=render_func,
                    enable_2pass=(planar_groups is not None and len(planar_groups) > 0),
                    planar_groups=planar_groups,
                    lambda_weight=args.lambda_weight,
                    metal_map=metal_map,
                    render_pkg_base=base_pkg,
                )
                if 'pass2' in pkg:
                    save_img_u8(pkg['pass2'].detach().permute(1,2,0).cpu().numpy(),
                                os.path.join(twopass_path, f'{stem}.png'))
                if 'specular_mask' in pkg:
                    m = pkg['specular_mask'].detach().expand(3,-1,-1)
                    save_img_u8(m.permute(1,2,0).cpu().numpy(),
                                os.path.join(planar_path, f'{stem}.png'))

    if not args.skip_train:
        print("export training images ...")
        _run_camera_set(scene.getTrainCameras(), train_dir)

    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        _run_camera_set(scene.getTestCameras(), test_dir)
    
    
    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name='render_traj', 
                    num_frames=n_fames)

