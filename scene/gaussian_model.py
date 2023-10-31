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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
from torch.nn import functional as F
import os
from typing import Any
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from knn_cuda import KNN
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from pytorch3d.transforms import quaternion_to_matrix, quaternion_invert

from .poly_taichi import Polynomial_taichi
from .fft_taichi import FFT_taichi

from utils.loss_utils import l1_loss, ssim, l2_loss

@torch.compile(backend="inductor", fullgraph=True)
def _rig_loss(tlast_rotation, tnow_rotation, dist_weight, tlast_dist, tnow_dist):
    # import pdb; pdb.set_trace()
    rigid_loss = dist_weight * (
        tlast_dist - (
            (
                quaternion_to_matrix(
                    tlast_rotation
                ) @ torch.inverse(
                    quaternion_to_matrix(
                        tnow_rotation
                    )
                )
            ).unsqueeze(1) @ tnow_dist.unsqueeze(-1)
        ).squeeze(-1)
    ).pow(2).sum(-1, True)
    return rigid_loss

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(
        self, 
        sh_degree : int,
        max_steps: int = 40_000,
        xyz_traj_feat_dim: int = 3, 
        rot_traj_feat_dim: int = 2, 
        scale_traj_feat_dim: int = 3,
        feature_traj_feat_dim: int = 2,
        xyz_trajectory_type: str = "fft", 
        rot_trajectory_type: str = "fft", 
        scale_trajectory_type : str = 'fft', 
        feature_trajectory_type: str = "fft",
        traj_init: str = "random",
        max_frames: Any = None, ):
        
        self.max_steps = max_steps
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        self._xyz_poly_params = torch.empty(0)
        self._rot_poly_params = torch.empty(0)
        self.setup_functions()
        
        self.xyz_traj_feat_dim = xyz_traj_feat_dim
        self.rot_traj_feat_dim = rot_traj_feat_dim
        self.scale_traj_feat_dim = scale_traj_feat_dim
        self.feature_traj_feat_dim = feature_traj_feat_dim
        
        self.xyz_trajectory_type = xyz_trajectory_type
        self.rot_trajectory_type = rot_trajectory_type
        self.scale_trajectory_type = scale_trajectory_type
        self.feature_trajectory_type = feature_trajectory_type
        self.xyz_trajectory_func = FFT_taichi(xyz_traj_feat_dim) if xyz_trajectory_type == "fft" else Polynomial_taichi(xyz_traj_feat_dim)
        self.rot_trajectory_func = FFT_taichi(rot_traj_feat_dim) if rot_trajectory_type == "fft" else Polynomial_taichi(rot_traj_feat_dim)
        # self.scale_trajectory_func = FFT_taichi(scale_traj_feat_dim) if scale_trajectory_type == "fft" else Polynomial_taichi(scale_traj_feat_dim)
        self.feature_trajectory_func = FFT_taichi(feature_traj_feat_dim) if feature_trajectory_type == "fft" else Polynomial_taichi(feature_traj_feat_dim)
        
        self.traj_init = torch.randn if traj_init == "random" else torch.zeros
        self.traj_fit_degree = 100000
        self.timestamp = 0.0
        self.max_frames = max_frames - 1 if max_frames else None
        
        self.ktop = 20
        self.knn = KNN(k=self.ktop, transpose_mode=True)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._xyz_poly_params,
            self._rot_poly_params,
            self._features_dc_poly_params,
            self._feature_poly_params,
        )
        
    def restore_t0(self, path):
        # load t0 knn points distance
        numpy_file = os.path.join(path, "t0_ktop_dist.npy")
        weight_factor = 2000.0
        self.t0_dist = torch.tensor(np.load(numpy_file)).cuda()
        # (N, ktop, 1)
        self.t0_dist_norm = self.t0_dist.pow(2).sum(-1, True).detach()
        self.dist_weight = torch.exp(
            -weight_factor * self.t0_dist_norm.pow(2)
        ).detach()
        
        
    def restore_diff(self, path):
        # load the difference between the current frame and the last frame
        numpy_file = os.path.join(path, "diff_xyz.npy")
        self.diff_xyz = torch.tensor(np.load(numpy_file)).cuda()
        numpy_file = os.path.join(path, "diff_rotation.npy")
        self.diff_rotation = torch.tensor(np.load(numpy_file)).cuda() 
    
    
    def restore(self, model_args, training_args):
        # import pdb; pdb.set_trace()
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._xyz_poly_params,
        self._rot_poly_params,
        self._features_dc_poly_params,
        self._feature_poly_params,) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        
    def update_fft_degree(self):
        self.traj_fit_degree += 1
        
    def set_no_deform(self):
        self._fwd_xyz = self._xyz
        self._fwd_rot = self._rotation
        self._fwd_features_dc = self._features_dc
        self._fwd_feature = self._features_rest
        
    @torch.no_grad()
    def get_knn_index(self):
        print("Running KNN")
        _xyz = self._xyz.data.clone().detach()
        _xyz_norm = _xyz.pow(2).sum(-1)
        # import pdb; pdb.set_trace()
        # dist = _xyz_norm.unsqueeze(1) - _xyz_norm.unsqueeze(0)
        row = _xyz_norm.unsqueeze(1) 
        col = _xyz_norm.unsqueeze(0)
        chunk_size = 10000
        col_chunk_size = 10000
        top_index_list = []
        for i in range(0, _xyz.shape[0], chunk_size):
            # dist_chunk = row[i:i+chunk_size] - col
            col_dist_chunk = []
            for j in range(0, _xyz.shape[0], col_chunk_size):
                dist_chunk = row[i:i+chunk_size] - col[:, j:j+col_chunk_size]
                col_dist_chunk.append(dist_chunk)
            col_dist = torch.cat(col_dist_chunk, dim=1)
            top_dist_i, top_index_i = torch.topk(col_dist, self.ktop, dim=-1, largest=False)
            top_index_list.append(top_index_i)
            
        top_index = torch.cat(top_index_list, dim=0)
        self.ktop_index = top_index
        # dist = distCUDA2(_xyz, _xyz)
        # knn_index = self.knn(_xyz, _xyz)
        # self.ktop_index = knn_index

    def knn_losses(self):
        _xyz_poly_params = self._xyz_poly_params
        _rot_poly_params = self._rot_poly_params
        
        B = _xyz_poly_params.shape[0]
        dynamic_threshold = 0.002
        moving_sum = torch.abs(_xyz_poly_params.reshape(B, -1)).sum(dim=-1)
        mask = moving_sum > dynamic_threshold
        
        
        ktop_index_mask = self.ktop_index[mask]
        
        if len(ktop_index_mask) > 0:
            ktop_fwd_xyz = _xyz_poly_params[ktop_index_mask].view(
                -1, self.ktop, *_xyz_poly_params.shape[1:]
            )
            
            ktop_fwd_rot = _rot_poly_params[ktop_index_mask].view(
                -1, self.ktop, *_rot_poly_params.shape[1:]
            )
        
            knn_xyz_loss = l1_loss(ktop_fwd_xyz, _xyz_poly_params[mask].unsqueeze(1))
            knn_rot_loss = l1_loss(ktop_fwd_rot, _rot_poly_params[mask].unsqueeze(1))
        else:
            knn_xyz_loss = 0.
            knn_rot_loss = 0.
            
        # print("moving_sum: ", moving_sum)
        # print("len(ktop_index_mask) :", len(ktop_index_mask) )
        return knn_xyz_loss + knn_rot_loss
        
        
    def set_timestamp(self, t, training=False, training_step=0, get_smooth_loss=False, use_interpolation=False, random_noise=False, get_moving_loss=False):
        
        if self.max_frames:
            self.timestamp = t/self.max_frames 
            offset_width = (1/self.max_frames)
        else:
            self.timestamp = t
            offset_width = 0.001
        losses = {}
        
        if random_noise and training and t != 0 and t != self.max_frames:
            noise_weight = 0.1 * (1 - (training_step/self.max_steps))
            self.timestamp += noise_weight*np.random.randn()
        
        mid_xyz = self.xyz_trajectory_func(
            self._xyz_poly_params.contiguous(), 
            self.timestamp, 
            self.traj_fit_degree
        )
        
        mid_rot = self.rot_trajectory_func(
            self._rot_poly_params.contiguous(), 
            self.timestamp, 
            self.traj_fit_degree
        )
        self.mid_xyz = mid_xyz
        self.mid_rot = mid_rot
        
        if training:
            if get_smooth_loss:
                neg_rand_offset = offset_width*0.5 if t != 0 else 0
                pos_rand_offset = offset_width*0.5 if t != self.max_frames else 0
                dist = neg_rand_offset + pos_rand_offset
                
                weight2 = neg_rand_offset
                weight1 = pos_rand_offset
                # add random noise 
                neg_xyz = self.xyz_trajectory_func(
                    self._xyz_poly_params.contiguous(), 
                    self.timestamp-neg_rand_offset, 
                    self.traj_fit_degree
                )
                pos_xyz = self.xyz_trajectory_func(
                    self._xyz_poly_params.contiguous(), 
                    self.timestamp+pos_rand_offset, 
                    self.traj_fit_degree
                )
                
                neg_rot = self.rot_trajectory_func(
                    self._rot_poly_params.contiguous(), 
                    self.timestamp-neg_rand_offset, 
                    # noise_rot,
                    self.traj_fit_degree
                )
                pos_rot = self.rot_trajectory_func(
                    self._rot_poly_params.contiguous(), 
                    self.timestamp+pos_rand_offset, 
                    # noise_rot,
                    self.traj_fit_degree
                )
                
                if use_interpolation:
                    self._fwd_xyz = self._xyz + (weight1*neg_xyz + weight2*pos_xyz) / dist
                    self._fwd_rot = self._rotation + (weight1*neg_rot + weight2*pos_rot) / dist
  
                smooth_loss_xyz = l1_loss(neg_xyz, mid_xyz) + l1_loss(pos_xyz, mid_xyz)
                smooth_loss_rot = l1_loss(neg_rot, mid_rot) + l1_loss(pos_rot, mid_rot)
                smooth_loss = smooth_loss_xyz + smooth_loss_rot
                losses.update({"smooth_loss": smooth_loss})

        if get_moving_loss:
            xyz_mean = torch.abs(self._xyz_poly_params).mean()
            rot_mean = torch.abs(self._rot_poly_params).mean()
            # feat_mean = self._features_dc_poly_params.mean()
            losses.update({"moving_loss": xyz_mean + rot_mean})
    
    
        self._fwd_xyz = self._xyz + mid_xyz
        self._fwd_rot = self._rotation + mid_rot
        
        # print("self._xyz: ", self._xyz)
        # print("mid_xyz: ", mid_xyz)
        # print("timestamp: ", self.timestamp)
            
        # self._fwd_scale = self._scaling + self.scale_trajectory_func(
        #     self._scale_poly_params.contiguous(), 
        #     self.timestamp, 
        #     self.traj_fit_degree
        # )
        # self._fwd_features_dc = self._features_dc + self.feature_trajectory_func(
        #     self._features_dc_poly_params.contiguous(), 
        #     self.timestamp, 
        #     self.traj_fit_degree
        # ).reshape(self._features_dc.shape)
        
        self._fwd_features_dc = self._features_dc

        # self._fwd_feature = self._features_rest
        self._fwd_feature = self._features_rest + self.feature_trajectory_func(
            self._feature_poly_params.contiguous(), 
            self.timestamp, 
            self.traj_fit_degree
        ).reshape(self._features_rest.shape)
        
        # if training_step > 20000:
        #     self._fwd_features_dc = self._features_dc.detach()
        #     self._fwd_feature = self._features_rest.detach()
        # else:
        #     self._fwd_features_dc = self._features_dc
        #     self._fwd_feature = self._features_rest
            
        return losses
        
            
        # if training:
        #     # add random noise 
        #     noise_weight = 0.01 * (1 - (training_step/self.max_steps))
        #     self._fwd_xyz += torch.randn_like(self._xyz) * noise_weight
        #     self._fwd_rot += torch.randn_like(self._rotation) * noise_weight
        #     self._fwd_features_dc += torch.randn_like(self._features_dc) * noise_weight
        #     self._fwd_feature += torch.randn_like(self._features_rest) * noise_weight

    # def fwd_xyz(self, t):
    #     self._fwd_xyz = self._xyz + self.xyz_trajectory_func(
    #         self._xyz_poly_params.contiguous(), t, 
    #         self.traj_fit_degree
    #     )
    #     # if t != 0:
    #     #     self._fwd_xyz = self._xyz + self.trajectory_func(self._xyz_poly_params, t/149)
    #     # else:
    #     #     self._fwd_xyz = self._xyz
        
    # def fwd_rot(self, t):
    #     self._fwd_rot = self._rotation + self.rot_trajectory_func(
    #         self._rot_poly_params.contiguous(), t, 

    #         self.traj_fit_degree
    #     )
        # if t != 0:
        #     self._fwd_rot = self._rotation + self.trajectory_func(self._rot_poly_params, t/149)
        # else:
        #     self._fwd_rot = self._rotation

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._fwd_rot)
    
    @property
    def get_xyz(self):
        return self._fwd_xyz
    
    @property
    def get_features(self):
        features_dc = self._fwd_features_dc
        features_rest = self._fwd_feature
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._fwd_rot)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        if self.xyz_trajectory_type == "fft":
            xyz_poly_params = self.traj_init((fused_point_cloud.shape[0], self.xyz_traj_feat_dim, 3, 2), device="cuda")
        elif self.xyz_trajectory_type == "poly":
            xyz_poly_params = self.traj_init((fused_point_cloud.shape[0], self.xyz_traj_feat_dim, 3), device="cuda")
            
        if self.rot_trajectory_type == "fft":
            rot_poly_params = torch.randn((fused_point_cloud.shape[0], self.rot_traj_feat_dim, 4, 2), device="cuda")
        elif self.rot_trajectory_type == "poly":
            rot_poly_params = torch.randn((fused_point_cloud.shape[0], self.rot_traj_feat_dim, 4), device="cuda")
    
        f_flat_dim = 3
        if self.feature_trajectory_type == "fft":
            features_dc_poly_params = self.traj_init((fused_point_cloud.shape[0], self.feature_traj_feat_dim, f_flat_dim, 2), device="cuda")
        elif self.feature_trajectory_type == "poly":
            features_dc_poly_params = self.traj_init((fused_point_cloud.shape[0], self.feature_traj_feat_dim, f_flat_dim), device="cuda")        
            
        f_flat_dim = (features.shape[-1]-1)*3
        if self.feature_trajectory_type == "fft":
            feature_poly_params = self.traj_init((fused_point_cloud.shape[0], self.feature_traj_feat_dim, f_flat_dim, 2), device="cuda")
        elif self.feature_trajectory_type == "poly":
            feature_poly_params = self.traj_init((fused_point_cloud.shape[0], self.feature_traj_feat_dim, f_flat_dim), device="cuda")
            
        # if self.scale_trajectory_type == "fft":
        #     scale_poly_params = self.traj_init((fused_point_cloud.shape[0], self.scale_traj_feat_dim, 3, 2), device="cuda")
        # elif self.scale_trajectory_type == "poly":
        #     scale_poly_params = self.traj_init((fused_point_cloud.shape[0], self.scale_traj_feat_dim, 3), device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        
        self._xyz_poly_params = nn.Parameter(xyz_poly_params.contiguous().requires_grad_(True))
        self._rot_poly_params = nn.Parameter(rot_poly_params.contiguous().requires_grad_(True))
        # self._scale_poly_params = nn.Parameter(scale_poly_params.contiguous().requires_grad_(True))
        self._feature_poly_params = nn.Parameter(feature_poly_params.contiguous().requires_grad_(True))
        self._features_dc_poly_params = nn.Parameter(features_dc_poly_params.contiguous().requires_grad_(True))
        
        self.fused_point_cloud = fused_point_cloud.cpu().clone().detach()
        self.features = features.cpu().clone().detach()
        self.scales = scales.cpu().clone().detach()
        self.rots = rots.cpu().clone().detach()
        self.opacities = opacities.cpu().clone().detach()
        
        self.xyz_poly_params = xyz_poly_params.cpu().clone().detach()
        self.rot_poly_params = rot_poly_params.cpu().clone().detach()
        # self.scale_poly_params = scale_poly_params.cpu().clone().detach()
        self.feature_poly_params = feature_poly_params.cpu().clone().detach()
        self.features_dc_poly_params = features_dc_poly_params.cpu().clone().detach()
        
    def reset_state(self):
        self._xyz = nn.Parameter(self.fused_point_cloud.clone().requires_grad_(True).to("cuda"))
        self._features_dc = nn.Parameter(self.features[:,:,0:1].clone().transpose(1, 2).contiguous().requires_grad_(True).to("cuda"))
        self._features_rest = nn.Parameter(self.features[:,:,1:].clone().transpose(1, 2).contiguous().requires_grad_(True).to("cuda"))
        self._scaling = nn.Parameter(self.scales.clone().requires_grad_(True).to("cuda"))
        self._rotation = nn.Parameter(self.rots.clone().requires_grad_(True).to("cuda"))
        self._opacity = nn.Parameter(self.opacities.clone().requires_grad_(True).to("cuda"))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self._xyz_poly_params = nn.Parameter(self.xyz_poly_params.clone().requires_grad_(True).to("cuda"))
        self._rot_poly_params = nn.Parameter(self.rot_poly_params.clone().requires_grad_(True).to("cuda"))
        # self._scale_poly_params = nn.Parameter(self.scale_poly_params.clone().requires_grad_(True).to("cuda"))
        

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._xyz_poly_params], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz_poly_params"},
            {'params': [self._rot_poly_params], 'lr': training_args.rotation_lr, "name": "rot_poly_params"},
            # {'params': [self._scale_poly_params], 'lr': training_args.scaling_lr, "name": "scale_poly_params"},
            {'params': [self._features_dc_poly_params], 'lr': training_args.feature_lr, "name": "features_dc_poly_params"},
            {'params': [self._feature_poly_params], 'lr': training_args.feature_lr / 20.0, "name": "feature_poly_params"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init*self.spatial_lr_scale,
            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )
        
        self.rot_scheduler_args = get_expon_lr_func(
            lr_init=training_args.rotation_lr,
            lr_final=training_args.rotation_lr*0.1,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            if param_group["name"] == "xyz_poly_params":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            # elif param_group["name"] == "rotation" or param_group["name"] == "rot_poly_params":
            #     lr = self.rot_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_t0(self, path):
        ktop = 20
        _xyz = self._xyz.data.clone().detach()
        knn_index = knn(_xyz, _xyz, ktop)
        ktop_index = knn_index[1]
        ktop_xyz = _xyz[ktop_index].view(
            -1, ktop, _xyz.shape[-1]
        )
        dist = ktop_xyz - _xyz.unsqueeze(1)
        # save as numpy file
        np.save(
            os.path.join(path, "t0_ktop_dist.npy"), 
            dist.detach().cpu().numpy()
        )
        
    def save_diff(self, path):
        # save the difference between the current frame and the last frame
        _xyz = self._xyz.data.clone().detach()
        _rot = self._rotation.data.clone().detach()
        diff_xyz = _xyz - self.tlast_xyz
        diff_rotation = _rot - self.tlast_rotation
        np.save(
            os.path.join(path, "diff_xyz.npy"), 
            diff_xyz.detach().cpu().numpy()
        )
        np.save(
            os.path.join(path, "diff_rotation.npy"), 
            diff_rotation.detach().cpu().numpy()
        )

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
                      
        xyz = self._xyz.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
            

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.1))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        

    def load_ply_for_rendering(self, path, only_xyz=False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        if not only_xyz:
            opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

            scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
            scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
                
            self._features_dc = torch.tensor(features_dc, dtype=torch.float, device="cpu").transpose(1, 2).contiguous()
            self._features_rest = torch.tensor(features_extra, dtype=torch.float, device="cpu").transpose(1, 2).contiguous()
            self._opacity = torch.tensor(opacities, dtype=torch.float, device="cpu")
            self._scaling = torch.tensor(scales, dtype=torch.float, device="cpu")

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.tensor(xyz, dtype=torch.float, device="cpu")
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cpu")

        self.active_sh_degree = self.max_sh_degree
        
    def to(self, device='cpu'):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        self._features_rest = self._features_rest.to(device)
        self._opacity = self._opacity.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._xyz_poly_params = optimizable_tensors["xyz_poly_params"]
        self._rot_poly_params = optimizable_tensors["rot_poly_params"]
        # self._scale_poly_params = optimizable_tensors["scale_poly_params"]
        self._features_dc_poly_params = optimizable_tensors["features_dc_poly_params"]
        self._feature_poly_params = optimizable_tensors["feature_poly_params"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self, 
        new_xyz, 
        new_features_dc, 
        new_features_rest, 
        new_opacities, 
        new_scaling, 
        new_rotation, 
        new_xyz_poly_params=None, 
        new_rot_poly_params=None,
        new_features_dc_poly_params=None,
        new_feature_poly_params=None,
        new_scale_poly_params=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation,
            "rot_poly_params" : new_rot_poly_params,
            "xyz_poly_params" : new_xyz_poly_params,
            # "scale_poly_params" : new_scale_poly_params,
            "features_dc_poly_params": new_features_dc_poly_params,
            "feature_poly_params" : new_feature_poly_params,
            
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._xyz_poly_params = optimizable_tensors["xyz_poly_params"]
        self._rot_poly_params = optimizable_tensors["rot_poly_params"]
        # self._scale_poly_params = optimizable_tensors["scale_poly_params"]
        self._features_dc_poly_params = optimizable_tensors["features_dc_poly_params"]
        self._feature_poly_params = optimizable_tensors["feature_poly_params"]
        
        # update fwd scaling
        # self._fwd_scale = self._scaling + self.scale_trajectory_func(
        #     self._scale_poly_params.contiguous(), 
        #     self.timestamp, 
        #     self.traj_fit_degree
        # )

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        if self.xyz_trajectory_type == "fft":
            new_xyz_poly_params = self._xyz_poly_params[selected_pts_mask].repeat(N,1,1, 1)
        elif self.xyz_trajectory_type == "poly":
            new_xyz_poly_params = self._xyz_poly_params[selected_pts_mask].repeat(N,1,1)
        
        if self.rot_trajectory_type == "fft":
            new_rot_poly_params = self._rot_poly_params[selected_pts_mask].repeat(N,1,1, 1)
        elif self.rot_trajectory_type == "poly":
            new_rot_poly_params = self._rot_poly_params[selected_pts_mask].repeat(N,1,1)
            
        # if self.scale_trajectory_type == "fft":
        #     new_scale_poly_params = self._scale_poly_params[selected_pts_mask].repeat(N,1,1, 1)
        # elif self.scale_trajectory_type == "poly":
        #     new_scale_poly_params = self._scale_poly_params[selected_pts_mask].repeat(N,1,1)
        
        if self.feature_trajectory_type == "fft":
            new_features_dc_poly_params = self._features_dc_poly_params[selected_pts_mask].repeat(N,1,1, 1)
            new_feature_poly_params = self._feature_poly_params[selected_pts_mask].repeat(N,1,1, 1)
        elif self.feature_trajectory_type == "poly":
            new_features_dc_poly_params = self._features_dc_poly_params[selected_pts_mask].repeat(N,1,1)
            new_feature_poly_params = self._feature_poly_params[selected_pts_mask].repeat(N,1,1)

        self.densification_postfix(
            new_xyz, 
            new_features_dc, 
            new_features_rest, 
            new_opacity, 
            new_scaling, 
            new_rotation, 
            new_xyz_poly_params, 
            new_rot_poly_params,
            # new_scale_poly_params,
            new_features_dc_poly_params,
            new_feature_poly_params,
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        new_xyz_poly_params = self._xyz_poly_params[selected_pts_mask]
        new_rot_poly_params = self._rot_poly_params[selected_pts_mask]
        # new_scale_poly_params = self._scale_poly_params[selected_pts_mask]
        new_features_dc_poly_params = self._features_dc_poly_params[selected_pts_mask]
        new_feature_poly_params = self._feature_poly_params[selected_pts_mask]

        self.densification_postfix(
            new_xyz, 
            new_features_dc, 
            new_features_rest, 
            new_opacities, 
            new_scaling, 
            new_rotation, 
            new_xyz_poly_params, 
            new_rot_poly_params,
            # new_scale_poly_params,
            new_features_dc_poly_params,
            new_feature_poly_params,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
    @torch.compile(backend="inductor")
    def regularization_losses(self):
        
        # rigid loss
        ktop_index = self.knn_index[1]
        _xyz = self.get_xyz
        xyz_ktop = _xyz[ktop_index].view(
            -1, self.ktop, _xyz.shape[-1]
        )
        tnow_dist = xyz_ktop - _xyz.unsqueeze(1)
        
        # tnow_dist_norm = tnow_dist.pow(2).sum(-1, True).sqrt()
        rig_loss = _rig_loss(
            self.tlast_rotation, 
            self.get_rotation,
            self.dist_weight,
            self.tlast_dist,
            tnow_dist,
        )
        # import pdb; pdb.set_trace()
        # rigid_loss = self.dist_weight * (
        #     self.tlast_dist - tnow_dist_comp
        # ).pow(2).sum(-1, True)
        
        # rigid_loss = self.tlast_dist - 
        
        # isometry loss
        # iso_loss = self.dist_weight * (
        #     self.t0_dist_norm - tnow_dist_norm
        # )
        # import pdb; pdb.set_trace()
        
        # rotation loss
        # tnow_rot_ktop = self._rotation[ktop_index].view(
        #     -1, self.ktop, self._rotation.shape[-1]
        # )
        # tlast_rot_ktop = self.tlast_rotation[ktop_index].view(
        #     -1, self.ktop, self.tlast_rotation.shape[-1]
        # )
        
        # tnow_rot_ktop @ 
        
        # return (rigid_loss + iso_loss).mean()
        return rig_loss.mean()
        
        