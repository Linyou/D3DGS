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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.use_extra_cam_info = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.load_img_factor = 1.0
        self.real_dynamic = False
        self.dataset_shuffle = False
        self.use_ensure_unique_sample = False
        self.aug_frist_end = False
        self.batch_on_t = False
        self.train_rest_frame = False
        self.after_second_frame = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init =  0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.005
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.prune_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.min_opacity = 0.005
        self.batch_size=1
        self.knn_loss = False
        self.scale_loss = False
        self.dataloader=False
        self.no_deform_from_iter=0
        self.factor_t = False
        self.offset_t = False
        self.factor_t_value = 0.5
        self.offset_t_value = 0.5
        self.loader_shuffle = False
        self.detach_base_iter = 5000
        self.opacity_mask = False
        self.normalize_timestamp = True
        super().__init__(parser, "Optimization Parameters")
        
class FlowParams(ParamGroup):
    def __init__(self, parser):
        self.xyz_traj_feat_dim = 16
        self.xyz_trajectory_type = 'poly'
        self.rot_traj_feat_dim = 20
        self.rot_trajectory_type = 'fft'
        self.scale_traj_feat_dim = 20
        self.scale_trajectory_type = 'none'
        self.opc_traj_feat_dim = 20
        self.opc_trajectory_type = 'none'
        self.feature_traj_feat_dim = 2
        self.feature_trajectory_type = 'none'
        self.feature_dc_trajectory_type = 'none'
        self.traj_init = 'zero'
        self.poly_base_factor = 1.0
        self.Hz_base_factor = 1.0
        self.normliaze = False
        self.get_smooth_loss=False
        self.use_interpolation=False
        self.random_noise=False
        self.get_moving_loss=False
        self.masked=False
        #lr
        # self.xyz_lr = 0.0001
        # self.rot_lr = 0.0001
        # self.scale_lr = 0.0001
        # self.opc_lr = 0.0001
        # self.feature_lr = 0.0001
        super().__init__(parser, "Gaussion Flow Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
