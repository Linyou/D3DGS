import nerfvis
from scene import Scene, GaussianModel
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import sys
from random import randint

import torch
import numpy as np

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

dataset = lp.extract(args)
opt = op.extract(args)
pipe = pp.extract(args)

gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians)

(model_params, first_iter) = torch.load(args.start_checkpoint)
gaussians.restore(model_params, opt)

viewpoint_stack = scene.getTrainCameras().copy()
viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
# prepare camera poses visualization
print(f"prepare camera poses visualization")
    
vis = nerfvis.Scene("nerf")
focal = fov2focal(
    viewpoint_cam.FoVy, 
    viewpoint_cam.image_height
)
w2c_mats = []
bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
# import pdb; pdb.set_trace()
for ii in range(len(viewpoint_stack)):
    cam_i = viewpoint_stack[ii]
    W2C = getWorld2View2(cam_i.R, cam_i.T)
    C2W = np.linalg.inv(W2C)
    w2c_mats.append(C2W)
w2c_mats = np.stack(w2c_mats, axis=0)
vis.remove("train_camera")
vis.add_camera_frustum("train_camera", focal_length=focal, image_width=viewpoint_cam.image_width, image_height=viewpoint_cam.image_height, z=0.3, r=w2c_mats[:, :3, :3], t=w2c_mats[:, :3, -1])

origin = np.array([0, 0, 0])
xp = np.array([1, 0, 0])
yp = np.array([0, 1, 0])
zp = np.array([0, 0, 1])
vis.add_line("xp", origin, xp, color=xp)
vis.add_line("yp", origin, yp, color=yp)
vis.add_line("zp", origin, zp, color=zp)

pc_color = gaussians.get_features[:, 0] * 0.2 + 0.5
dist = gaussians.get_xyz.pow(2).sum(dim=-1).sqrt()
dist_mask = dist < 20.0
mask_xyz = gaussians.get_xyz[dist_mask]
mask_xyz *= torch.tensor([[1, -1, -1]], device=mask_xyz.device) 
# mask_xyz -= torch.tensor([[1, 0, 0]], device=mask_xyz.device)
mask_color = pc_color[dist_mask]
# import pdb; pdb.set_trace()
vis.remove("pc")
vis.add_points("pc", mask_xyz.cpu().detach().numpy(), vert_color=mask_color.cpu().detach().numpy())
vis.display(port=8889, serve_nonblocking=False)

# python vis.py -s /home/loyot/workspace/SSD_1T/Datasets/NeRF/yundong --start_checkpoint output/yundong_base/chkpnt30000.pth --model_path output/yundong_base/