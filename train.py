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

import os
from copy import deepcopy
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DynamicScene
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, FlowParams

from render_dynamic import render_sets

import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from lpips import LPIPS

from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
    
import taichi as ti
ti.init(arch=ti.cuda, offline_cache=False)

lpips_net = LPIPS(net="vgg").to("cuda")
lpips_norm_fn = lambda x: x[None, ...] * 2 - 1
lpips_norm_b_fn = lambda x: x * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()
lpips_b_fn = lambda x, y: lpips_net(lpips_norm_b_fn(x), lpips_norm_b_fn(y)).mean()

def training(dataset, opt, pipe, flow_args, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, smc_file=None):
    
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.sh_degree,
        max_steps=opt.iterations+1,
        xyz_traj_feat_dim=flow_args.xyz_traj_feat_dim,
        xyz_trajectory_type=flow_args.xyz_trajectory_type,
        rot_traj_feat_dim=flow_args.rot_traj_feat_dim,
        rot_trajectory_type=flow_args.rot_trajectory_type,
        scale_traj_feat_dim=flow_args.scale_traj_feat_dim,
        scale_trajectory_type=flow_args.scale_trajectory_type,
        opc_traj_feat_dim=flow_args.opc_traj_feat_dim,
        opc_trajectory_type=flow_args.opc_trajectory_type,
        feature_traj_feat_dim=flow_args.feature_traj_feat_dim,
        feature_trajectory_type=flow_args.feature_trajectory_type,
        feature_dc_trajectory_type=flow_args.feature_dc_trajectory_type,
        traj_init=flow_args.traj_init,
        poly_base_factor=flow_args.poly_base_factor,
        Hz_base_factor=flow_args.Hz_base_factor,
        normliaze=flow_args.normliaze,
        factor_t=opt.factor_t,
        factor_t_value=opt.factor_t_value,
        offset_t=opt.offset_t,
        offset_t_value=opt.offset_t_value,
        normalize_timestamp=opt.normalize_timestamp,
    )
    if pipe.real_dynamic:
        scene = DynamicScene(dataset, gaussians)
    else:
        scene = Scene(
            dataset, 
            gaussians, 
            shuffle=pipe.dataset_shuffle, 
            load_img_factor=pipe.load_img_factor,
            smc_file=smc_file,
        )
        
    gaussians.training_setup(opt, flow_args)
    # if opt.train_rest_frame:
    #     gaussians.fix_params_rest_of_frames()
    frame_id = scene.model_path.split("/")[-1]
    model_root = scene.model_path.replace('/'+frame_id, '/')
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt, flow_args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    time_view_stack = None
    ema_loss_for_log = 0.0
    
    frist_update_moving_mask = True
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    knn_loss = None
    scale_loss = None
    moving_loss = None
    smooth_loss = None
    Ll2 = None
    if opt.knn_loss:
        gaussians.get_knn_index()
        
    # grad_scaler = torch.cuda.amp.GradScaler(2**10)
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 100 == 0:
            gaussians.oneupSHdegree()
            
        if iteration == opt.no_deform_from_iter:
            viewpoint_stack = None
            
        # Pick a random Camera
        if not viewpoint_stack:
            if iteration < opt.no_deform_from_iter:
                if scene.train_cameras_0 is not None:
                    viewpoint_stack= scene.getTrainCameras_0()
                else:
                    viewpoint_stack = scene.getTrainCameras()
            else:
                viewpoint_stack = scene.getTrainCameras()
            batch_size = opt.batch_size
            if opt.dataloader:
                viewpoint_stack_loader = DataLoader(
                    viewpoint_stack, 
                    batch_size=batch_size,
                    shuffle=opt.loader_shuffle,
                    num_workers=16,
                    collate_fn=list
                )
                loader = iter(viewpoint_stack_loader)
            else:
                viewpoint_stack = [i for i in viewpoint_stack]
                temp_list = deepcopy(viewpoint_stack)
                viewpoint_stack = temp_list.copy()
        if opt.dataloader:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                # print("reset dataloader")
                batch_size = 1
                loader = iter(viewpoint_stack_loader)
        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :    
                    
                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
            
        renders = []
        renders_opacities = []
        viewspace_points = []
        visibility_filters = []
        radiis = []
        gt_images = []
        gt_mask = []
        for viewpoint_cam in viewpoint_cams:
                
            time_sample = viewpoint_cam.timestamp
            # print(f"time_sample: {time_sample}")
            if iteration > opt.no_deform_from_iter:
                gaussians.need_deformed = True
                gaussians.set_timestamp(
                    time_sample, 
                    training=True, 
                    training_step=iteration, 
                    random_noise=flow_args.random_noise,
                    masked=flow_args.masked,
                    detach_base=iteration > opt.detach_base_iter,
                )
                # testing_iterations = 1
            else:
                gaussians.need_deformed = False
                gaussians.set_no_deform()
            
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            
            renders.append(render_pkg["render"])
            viewspace_points.append(render_pkg["viewspace_points"])
            visibility_filters.append(render_pkg["visibility_filter"].unsqueeze(0))
            radiis.append(render_pkg["radii"].unsqueeze(0))
            gt_images.append(viewpoint_cam.original_image.cuda())
            if opt.opacity_mask:
                renders_opacities.append(render_pkg["opacity"])
                # import pdb; pdb.set_trace()
                gt_mask.append(viewpoint_cam.extra_cam_info.mask.float().cuda())
            

        # Loss
        # gt_image = viewpoint_cam.original_image.cuda()
        radii = torch.cat(radiis,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filters).any(dim=0)
        images = torch.stack(renders)
        gt_images = torch.stack(gt_images)    
        

        Ll1 = l1_loss(images, gt_images)
        # Ll1 = F.smooth_l1_loss(images, gt_images)
        # Ll1 = l2_loss(images, gt_images) 
        # ssim_loss = ssim(images, gt_images)
        # lpips_loss = lpips_b_fn(images, gt_images)
        loss = Ll1
        # loss = Ll1 + 0.01 * (1.0 - ssim_loss) #+ 0.01 * lpips_loss
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(images, gt_images))
        
        if opt.mse_loss:
            Ll2 = l2_loss(images, gt_images)
            loss += Ll2
        
        if opt.opacity_mask:
            opacity = torch.stack(renders_opacities)
            gt_mask = torch.stack(gt_mask)
            mask_loss = l1_loss(opacity, gt_mask)
            # import pdb; pdb.set_trace()
            loss += mask_loss
        else:
            mask_loss = None
            
        if opt.knn_loss and iteration > opt.densify_until_iter:
            knn_loss = gaussians.knn_losses()
            loss += knn_loss * 0.1
            
        if flow_args.get_moving_loss:
            moving_loss = gaussians.moving_loss()
            loss += moving_loss
            
        if flow_args.get_smooth_loss:
            smooth_loss = gaussians.smooth_loss()
            loss += smooth_loss
            
        if pipe.train_rest_frame and (iteration > 0):
            reg_loss = gaussians.regularization_losses()
            loss += reg_loss
        else:
            reg_loss = None
            
        if opt.scale_loss:
            scale_loss = gaussians.scale_losses()
            loss += scale_loss
            
        if loss > 1000:
            continue
            
        loss.backward()
        # grad_scaler.scale(loss).backward()
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_points[0])
        for idx in range(0, len(viewspace_points)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_points[idx].grad
        # import pdb; pdb.set_trace()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                bar_info = {}
                if reg_loss is not None:
                    ema_reg_loss_for_log = 0.4 * reg_loss.item() + 0.6 * ema_loss_for_log
                    bar_info.update({"RegLoss": f"{ema_reg_loss_for_log:.{7}f}"})
                if knn_loss is not None:
                    bar_info.update({"KnnLoss": f"{knn_loss:.{7}f}"})
                if moving_loss is not None:
                    bar_info.update({"MoveLoss": f"{moving_loss:.{7}f}"})
                if smooth_loss is not None:
                    bar_info.update({"SmoothLoss": f"{smooth_loss:.{7}f}"})
                if scale_loss is not None:
                    bar_info.update({"ScaleLoss": f"{scale_loss:.{7}f}"})
                if mask_loss is not None:
                    bar_info.update({"MaskLoss": f"{mask_loss:.{7}f}"})
                if Ll2 is not None:
                    bar_info.update({"Ll2": f"{Ll2:.{7}f}"})
                bar_info.update({"Loss": f"{ema_loss_for_log:.{7}f}"})
                bar_info.update({"NumGass": f"{gaussians._xyz.shape[0]}"})
                # if iteration > 5000:
                #     bar_info.update({"Mask num": f"{gaussians.moving_mask.sum()}"})
                #     bar_info.update({"Mask threshold": f"{gaussians.threshold:.{5}f}"})
                progress_bar.set_postfix(bar_info)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if pipe.real_dynamic:
                training_report_real_dynamic(tb_writer, iteration, Ll1, loss, l1_loss, reg_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            else:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                
            if (iteration % saving_iterations) == 0:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:

                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                densification_interval = opt.densification_interval
                opc_reset_interval = opt.opacity_reset_interval
                densify_grad_threshold = opt.densify_grad_threshold
                min_opacity = opt.min_opacity
                prune_interval = opt.prune_interval
                # if iteration > 5000:
                #     densification_interval = 1000
                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    gaussians.densify(densify_grad_threshold, scene.cameras_extent)
                if iteration > opt.densify_from_iter and iteration % prune_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(min_opacity, scene.cameras_extent, size_threshold)
                    # if iteration > 5000:
                    #     gaussians.update_moving_mask(iteration)
                if iteration % opc_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                
            if iteration == opt.densify_until_iter and opt.knn_loss:
                gaussians.get_knn_index()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                # gaussians.optimizer_poly.step()
                # gaussians.optimizer_poly.zero_grad(set_to_none = True)

            if (iteration % checkpoint_iterations) == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
                # if not opt.train_rest_frame:
                #     gaussians.save_t0(scene.model_path.replace(frame_id, ''))
                # else:
                #     gaussians.save_diff(scene.model_path.replace(frame_id, ''))
        # torch.cuda.empty_cache()
    print("\n[ITER {}] Saving Checkpoint".format(iteration))
    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report_real_dynamic(tb_writer, iteration, Ll1, loss, l1_loss, reg_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        if reg_loss is not None:
            tb_writer.add_scalar('train_loss_patches/reg_loss', reg_loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration % testing_iterations == 0:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(0, len(scene.getTestCameras()), 10)]}, 
            {'name': 'train', 'cameras' : []}
        )
        

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # import pdb; pdb.set_trace()
                    scene.gaussians.set_timestamp(viewpoint.timestamp)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    tb_writer.add_images(config['name'] + "_view_{}_frame_{}/render".format(viewpoint.image_name, viewpoint.timestamp), image[None], global_step=iteration)
                    tb_writer.add_images(config['name'] + "_view_{}_frame_{}/ground_truth".format(viewpoint.image_name, viewpoint.timestamp), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssims_test += ms_ssim(
                        image[None], gt_image[None], data_range=1, size_average=True
                    )   
                    lpips_test += lpips_fn(image, gt_image).item()
                psnr_test /= len(config['cameras'])
                ssims_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.5f} PSNR {psnr_test:.5f} SSIMS {ssims_test:.5f} LPIPS {lpips_test:.5f}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssims', ssims_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)


    # Report test and samples of training set
    if iteration % testing_iterations == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})


        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    scene.gaussians.set_timestamp(viewpoint.timestamp)
                    render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # opacity = render_result["opacity"]
                    # depth = render_result["depth"]
                    # depth_normal = (depth - depth.min()) / (depth.max() - depth.min())
                    # gt_opacity = None
                    # if viewpoint.extra_cam_info is not None:
                    #     if viewpoint.extra_cam_info.mask is not None:
                    #         gt_opacity = viewpoint.extra_cam_info.mask.float().cuda()
                    if tb_writer:
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/opacity".format(viewpoint.image_name), opacity[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth_normal[None], global_step=iteration)
                        # if gt_opacity is not None:
                        #     tb_writer.add_images(config['name'] + "_view_{}/gt_opacity".format(viewpoint.image_name), gt_opacity[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssims_test += ms_ssim(
                        image[None], gt_image[None], data_range=1, size_average=True
                    )   
                    lpips_test += lpips_fn(image, gt_image).item()
                psnr_test /= len(config['cameras'])
                ssims_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.5f} PSNR {psnr_test:.5f} SSIMS {ssims_test:.5f} LPIPS {lpips_test:.5f}")
            if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssims', ssims_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)


        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    ff = FlowParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", type=int, default=5000)
    parser.add_argument("--save_iterations", type=int, default=10000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", type=int, default=30000)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--smc_file", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
        
        # copy config to output
        import shutil
        # get file name of the config
        os.makedirs(args.model_path)
        config_filename = args.configs.split('/')[-1]
        shutil.copyfile(
            args.configs, 
            os.path.join(args.model_path, config_filename),
        )
        
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        ff.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.smc_file,
    )
    max_iter = args.iterations
    print(f"Render video using chkpnt{max_iter}.pth")
    render_sets(
        lp.extract(args), 
        op.extract(args), 
        ff.extract(args), 
        max_iter, 
        pp.extract(args), 
        skip_train=True,
        skip_test=True,
        skip_video=False,
        save_npz=True,
        ckpt_name=f"chkpnt{max_iter}.pth",
    )

    # All done
    print("\nTraining complete.")
