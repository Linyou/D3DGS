import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import cv2

import time

import numpy as np
import torch
import torch.nn.functional as F

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import taichi as ti

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DynamicScene
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import datetime

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).cpu().numpy().astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img.astype(np.float32)

@ti.kernel
def write_buffer(
    reverse_h: bool,
    W:ti.i32, H:ti.i32, 
    x: ti.types.ndarray(), 
    final_pixel:ti.template()
):
    for i, j in ti.ndrange(W, H):
        j_rev = j
        if reverse_h:
            j_rev = H - j - 1
        for p in ti.static(range(3)):
            final_pixel[i, j][p] = x[p, j_rev, i]

class GUI:
    def __init__(self, dataset, opt, pipe, checkpoint, dynamic=False):

        device = "cuda:0"
        self.dynamic = dynamic

        self.gaussians_list = []
        self.current_gaussians = GaussianModel(dataset.sh_degree)
        if dynamic:
            self.scene = DynamicScene(
                dataset, 
                self.current_gaussians, 
                only_frist=True,
            )
        else:
            self.scene = Scene(dataset, self.current_gaussians)
        self.pipe = pipe
        self.current_gaussians.training_setup(opt)
        (model_params, first_iter) = torch.load(checkpoint)
        self.current_gaussians.restore(model_params, opt)
        # path_str = dataset.model_path
        # gaussians_paths_root = path_str.replace("/0", "")
        # spath = f"{gaussians_paths_root}/1/point_cloud/iteration_2000/point_cloud.ply"
        # self.current_gaussians.load_ply_for_rendering(spath)
        # self.current_gaussians.to("cuda")
        # for idx in tqdm(range(1, 300)):
        #     if idx == 0:
        #         spath = f"{gaussians_paths_root}/{idx}/point_cloud/iteration_10000/point_cloud.ply"
        #     else:
        #         spath = f"{gaussians_paths_root}/{idx}/point_cloud/iteration_2000/point_cloud.ply"
        #     gg = GaussianModel(dataset.sh_degree)
        #     gg.load_ply_for_rendering(spath, only_xyz=True)
        #     self.gaussians_list.append(gg)
            
        # import pdb; pdb.set_trace()
        # self.gaussians.training_setup(opt)
        
        # (model_params, first_iter) = torch.load(args.start_checkpoint)
        # self.gaussians.restore(model_params, opt)
        
        self.bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")

        if dynamic:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()[0]
        else:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
            
        self.viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
        # placeholders
        self.dt = 0
        self.mean_samples = 0
        self.img_mode = 0
        # import pdb; pdb.set_trace()

        self.iter_start = torch.cuda.Event(enable_timing = True)
        self.iter_end = torch.cuda.Event(enable_timing = True)
        self.H=int(self.viewpoint_cam.image_height)
        self.W=int(self.viewpoint_cam.image_width)

        self.camera = ti.ui.Camera()
        self.camera.position(1, 1, 1)
        self.camera.lookat(0, 0, 0)
        self.camera.up(0, 1, 0)
        
    @torch.no_grad()
    def render_frame(self):
        t = time.time()
        # print(cam.pose)
        with torch.no_grad():
            render_pkg = render(self.viewpoint_cam, self.current_gaussians, self.pipe, self.background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        self.dt = time.time()-t

        return image

    def render_gui(self):

        ti.init(arch=ti.cuda, offline_cache=True)

        W, H = self.W, self.H
        print("W:", type(W))
        final_pixel = ti.Vector.field(n=3, dtype=float, shape=(W, H))

        window = ti.ui.Window('Window Title', (W, H),)
        canvas = window.get_canvas()
        gui = window.get_gui()
        playing = False
        first_play = False

        current_frame = 0
        start = datetime.datetime.now()
        
        self.current_gaussians.fwd_xyz(current_frame)
        self.current_gaussians.fwd_rot(current_frame)
        
        num_pos = self.current_gaussians.get_xyz.shape[0]

        while window.running:
            self.camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

            with gui.sub_window("Options", 0.01, 0.01, 0.25, 0.28) as w:

                if gui.button('play'):
                    playing = True
                if gui.button('pause'):
                    playing = False
                    
                update_frame = False
                if playing:
                    end = datetime.datetime.now()
                    duration = (end - start).total_seconds() * 1000  # Convert to milliseconds

                    if duration >= 40:  # 25 fps
                        if not first_play:
                            current_frame += 1
                            if current_frame > 149:
                                current_frame = 0
                            update_frame = True
                            # print("Frame:", current_frame)  # Uncomment to print the frame number

                        else:
                            first_play = False

                        start = datetime.datetime.now()
                else:
                    first_play = True
                        
                # cam_pose = self.cam.pose
                w.text(f'render size: {self.viewpoint_cam.image_width} x {self.viewpoint_cam.image_height}')
                w.text(f'number of gaussians: {num_pos}')
                w.text(f'render times: {1000*self.dt:.2f} ms')
                w.text(f'c2w:')
                w.text(f'{self.viewpoint_cam.R[0]}')
                w.text(f'{self.viewpoint_cam.R[1]}')
                w.text(f'{self.viewpoint_cam.R[2]}')
                w.text('position:')
                w.text(
                    f'-{self.camera.curr_position}'
                )
                w.text('lookat:')
                w.text(
                    f'-{self.camera.curr_lookat}'
                )
                w.text('up:')
                w.text(
                    f'-{self.camera.curr_up}'
                )
            Rt = self.camera.get_view_matrix().T
            Rt[:3, 3] *= np.array([1, 1, -1])
            # t = np.array([
            #     -self.camera.curr_position[0], 
            #     self.camera.curr_position[1],
            #     self.camera.curr_position[2],
            # ])
            # self.viewpoint_cam.new_cam(self.cam.rot, self.cam.center)
            self.viewpoint_cam.new_cam(Rt[:3, :3], Rt[:3, 3])
            # print("frame id: ", current_frame)
            if update_frame:
                if self.dynamic:
                    self.current_gaussians.fwd_xyz(current_frame)
                    self.current_gaussians.fwd_rot(current_frame)
                else:
                    self.current_gaussians._xyz[...] = self.gaussians_list[current_frame]._xyz
                    self.current_gaussians._rotation[...] = self.gaussians_list[current_frame]._rotation
            render_buffer = self.render_frame()
            # print("render_buffer shape: ", render_buffer.shape)
            write_buffer(True, W, H, render_buffer, final_pixel)
            canvas.set_image(final_pixel)
            window.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[2_000, 10_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--dynamic", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    print(args.source_path)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    gui = GUI(dataset, opt, pipe, args.start_checkpoint, args.dynamic)
    gui.render_gui()
    # args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    
