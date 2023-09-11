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
from taichi.math import vec3, ivec2

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import nerfvis
from utils.sh_utils import SH2RGB
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

def prepare_output(args):    
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
    def __init__(self, dataset, opt, pipe):

        device = "cuda:0"
        prepare_output(dataset)
        self.gaussians = GaussianModel(dataset.sh_degree)
        self.scene = Scene(dataset, self.gaussians)
        self.gaussians.training_setup(opt)
        self.dataset = dataset
        self.pipe = pipe
        self.opt = opt
        
        self.bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")

        viewpoint_stack = self.scene.getTrainCameras().copy()
        self.viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        self.viewpoint_stack = self.scene.getTrainCameras().copy()
        # import pdb; pdb.set_trace()
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
        self.camera.position(0, 0, 2)
        self.camera.lookat(0, 0, 1)
        self.camera.up(0, 1, 0)
        self.camera.fov(
            np.rad2deg(self.viewpoint_cam.FoVx)
        )
        
        self.iteration = 0
        
        R_np = []
        T_np = []
        w2c_mats = []
        # import pdb; pdb.set_trace()
        for ii in range(len(viewpoint_stack)):
            cam_i = viewpoint_stack[ii]
            R_np.append(cam_i.R)
            T_np.append(cam_i.T)
            W2C = getWorld2View2(cam_i.R, cam_i.T)
            C2W = np.linalg.inv(W2C)
            w2c_mats.append(C2W)
        w2c_mats = np.stack(w2c_mats, axis=0)

        self.cam_T = w2c_mats[:, :3, -1]
        self.cam_pos = ti.Vector.field(
            3, dtype=ti.f32, 
            shape=len(viewpoint_stack)
        )
        for i in tqdm(range(len(viewpoint_stack))):
            self.cam_pos[i] = vec3(
                self.cam_T[i, 0],
                self.cam_T[i, 1],
                self.cam_T[i, 2],
            )
            
    @torch.no_grad()
    def render_frame(self):
        t = time.time()
        # print(cam.pose)
        with torch.no_grad():
            render_pkg = render(self.viewpoint_cam, self.gaussians, self.pipe, self.background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        self.dt = time.time()-t

        return image

    def render_gui(self):

        W, H = self.W, self.H
        print("W:", type(W))
        final_pixel = ti.Vector.field(n=3, dtype=float, shape=(W, H))

        window = ti.ui.Window('Window Title', (W, H),)
        canvas = window.get_canvas()
        gui = window.get_gui()
        training = True
        cur_cam_id = 0
        start_time = time.time()
        eta_times = 0
        num_pos = self.gaussians.get_xyz.shape[0]
        
        scene = ti.ui.Scene()
        pos_lines = ti.Vector.field(3, dtype=ti.f32, shape=(6, ))
        pos_lines_colors = ti.Vector.field(3, dtype=ti.f32, shape=(6, ))
        pos_lines[0] = vec3(0, 0, 0)
        pos_lines_colors[0] = vec3(1, 0, 0)
        pos_lines[1] = vec3(1, 0, 0)
        pos_lines_colors[1] = vec3(1, 0, 0)
        
        pos_lines[2] = vec3(0, 0, 0)
        pos_lines_colors[2] = vec3(0, 1, 0)
        pos_lines[3] = vec3(0, 1, 0)
        pos_lines_colors[3] = vec3(0, 1, 0)
        
        pos_lines[4] = vec3(0, 0, 0)
        pos_lines_colors[4] = vec3(0, 0, 1)
        pos_lines[5] = vec3(0, 0, 1)
        pos_lines_colors[5] = vec3(0, 0, 1)
        
        pos_vertex = ti.Vector.field(3, dtype=ti.f32, shape=(8, ))
        pos_indice = ti.Vector.field(2, dtype=ti.i32, shape=(12, ))
        vertex = [
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        for i in range(7):
            pos_vertex[i] = vec3(*vertex[i])
        indice = [
            [0, 6],
            [0, 5],
            [0, 1],
            [4, 1],
            [4, 5],
            [4, 3],
            [2, 6],
            [2, 3],
            [2, 1],
            [7, 6],
            [7, 5],
            [7, 3],
        ]
        for i in range(9):
            pos_indice[i] = ivec2(*indice[i])

        while window.running:
            self.camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
            scene.set_camera(self.camera)
            scene.ambient_light((0.8, 0.8, 0.8))

            with gui.sub_window("Options", 0.01, 0.01, 0.23, 0.6) as w:

                if self.iteration > 50000:
                    training = False
                    
                if gui.button('start training'):
                    start_time = time.time()
                    training = True
                if gui.button('pause training'):
                    training = False
                if gui.button('restart training'):
                    start_time = time.time()
                    self.gaussians.reset_state()
                    self.gaussians.training_setup(self.opt)
                    
                pos_x = gui.slider_float("pos_x", self.camera.curr_position[0], minimum=-5, maximum=5)
                pos_y = gui.slider_float("pos_y", self.camera.curr_position[1], minimum=-5, maximum=5)
                pos_z = gui.slider_float("pos_z", self.camera.curr_position[2], minimum=-5, maximum=5)
                self.camera.position(pos_x, pos_y, pos_z)
                
                look_x = gui.slider_float("look_x", self.camera.curr_lookat[0], minimum=-5, maximum=5)
                look_y = gui.slider_float("look_y", self.camera.curr_lookat[1], minimum=-5, maximum=5)
                look_z = gui.slider_float("look_z", self.camera.curr_lookat[2], minimum=-5, maximum=5)
                self.camera.lookat(look_x, look_y, look_z)
                
                up_x = gui.slider_int("up_x", self.camera.curr_up[0], minimum=-1, maximum=1)
                up_y = gui.slider_int("up_y", self.camera.curr_up[1], minimum=-1, maximum=1)
                up_z = gui.slider_int("up_z", self.camera.curr_up[2], minimum=-1, maximum=1)
                self.camera.up(up_x, up_y, up_z)       
                
                # cam_pose = self.cam.pose
                if training:
                    eta_times = time.time()-start_time
                
                w.text(f'render size: {self.viewpoint_cam.image_width} x {self.viewpoint_cam.image_height}')
                w.text(f'training time: {eta_times:.2f} sec')
                w.text(f'training iter: {self.iteration}')
                w.text(f'current training cam: {cur_cam_id}')
                if self.iteration % 10:
                    num_pos = self.gaussians.get_xyz.shape[0]
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
                
            if training:
                cur_cam_id = self.training_step()
                
            Rt = self.camera.get_view_matrix()
            R = Rt[:3, :3].T
            # R *= np.array([[1, 1, 1]])
            Rt[3, :3] *= np.array([1, -1, -1])
            # t = np.array([
            #     -self.camera.curr_position[0], 
            #     self.camera.curr_position[1],
            #     self.camera.curr_position[2],
            # ])
            # import pdb; pdb.set_trace()
            # self.viewpoint_cam.new_cam(self.cam.rot, self.cam.center)
            self.viewpoint_cam.new_cam(R[:, [0 ,2, 1]], Rt[3, :3])
            # print("frame id: ", current_frame)
            render_buffer = self.render_frame()
            # print("render_buffer shape: ", render_buffer.shape)
            write_buffer(True, W, H, render_buffer, final_pixel)
            canvas.set_image(final_pixel)
            
            # scene.particles(self.cam_pos, color=(1, 1, 1), radius=0.01)
            scene.lines(pos_vertex, indices=pos_indice, color=(0.28, 0.68, 0.99), width=5.0)
            scene.lines(pos_lines, per_vertex_color=pos_lines_colors, width=5.0)
            canvas.scene(scene)
            window.show()
            
            
    def training_step(self):
        # optimization
        self.gaussians.update_learning_rate(self.iteration)
        self.iteration += 1
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()
        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        cam_id = randint(0, len(self.viewpoint_stack)-1)
        viewpoint_cam = self.viewpoint_stack.pop(cam_id)
        # import pdb; pdb.set_trace()
        render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        
        with torch.no_grad():
            # Densification
            if self.iteration < self.opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.iteration > opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                
                if self.iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none = True)
                
        return cam_id
        
        

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
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    ti.init(arch=ti.cuda, offline_cache=True)
    
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
    gui = GUI(dataset, opt, pipe)
    gui.render_gui()
    # args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    
