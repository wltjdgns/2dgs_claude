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
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos, save_img_u8

import open3d as o3d

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (alias for --model_path)")
    # PBR / 2-pass options
    parser.add_argument("--enable_2pass", action="store_true",
                        help="Enable 2-pass planar reflection rendering (requires SAM)")
    parser.add_argument("--sam_ckpt", type=str, default=None,
                        help="Path to SAM checkpoint (e.g. sam_vit_h_4b8939.pth)")
    parser.add_argument("--sam_model_type", type=str, default="vit_h",
                        help="SAM model type: vit_h / vit_l / vit_b")
    parser.add_argument("--metalnet_ckpt", type=str, default=None,
                        help="Path to MetalicNet checkpoint for metal map inference")
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
    if args.enable_2pass and args.sam_ckpt:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_ckpt).cuda()
        sam_generator = SamAutomaticMaskGenerator(sam)
        print(f"[SAM] loaded {args.sam_model_type} from {args.sam_ckpt}")

    def render_func(cam, gs, pipe_, bg):
        return render(cam, gs, pipe_, bg)

    def _run_camera_set(cameras, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        gaussExtractor.reconstruction(cameras)
        gaussExtractor.export_image(out_dir)

        if not (args.enable_2pass or metalnet):
            return

        from utils.planar_utils import detect_planar_groups_from_depth_fast
        from gaussian_renderer.render_2pass import render_2pass
        vis_path = os.path.join(out_dir, "vis")
        os.makedirs(vis_path, exist_ok=True)

        for idx, cam in tqdm(enumerate(cameras), desc="PBR / 2-pass export"):
            stem = '{0:05d}'.format(idx)
            base_pkg = render(cam, gaussians, pipe, background)

            # ── SAM planar detection (per-frame, optional) ──
            planar_groups = None
            if sam_generator is not None:
                planar_groups = detect_planar_groups_from_depth_fast(
                    cam, gaussians, pipe, background, render_func,
                    use_sam=True, sam_generator=sam_generator,
                )

            # ── 2-pass rendering ──
            if args.enable_2pass:
                pkg = render_2pass(
                    cam, gaussians, pipe, background,
                    render_func=render_func,
                    enable_2pass=(planar_groups is not None and len(planar_groups) > 0),
                    planar_groups=planar_groups,
                )
                if 'render' in pkg and pkg['render'] is not pkg.get('pass1'):
                    save_img_u8(pkg['render'].permute(1,2,0).cpu().numpy(),
                                os.path.join(vis_path, f'2pass_{stem}.png'))
                if 'specular_mask' in pkg:
                    m = pkg['specular_mask'].expand(3,-1,-1)
                    save_img_u8(m.permute(1,2,0).cpu().numpy(),
                                os.path.join(vis_path, f'planar_mask_{stem}.png'))
                base_pkg = pkg   # use enriched pkg for metalnet below

            # ── MetalicNET ──
            if metalnet is not None:
                from utils.metalnet_utils import predict_metal_map, metalprob_to_f0_rgb
                metal = predict_metal_map(metalnet, base_pkg)
                if metal is not None:
                    save_img_u8(metal.expand(3,-1,-1).permute(1,2,0).cpu().numpy(),
                                os.path.join(vis_path, f'metal_{stem}.png'))
                    f0 = metalprob_to_f0_rgb(base_pkg, metal)
                    if f0 is not None:
                        save_img_u8(f0.permute(1,2,0).cpu().numpy(),
                                    os.path.join(vis_path, f'f0_{stem}.png'))

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

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))