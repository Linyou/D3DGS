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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

from tqdm import tqdm
from scene.dataset import FourDGSdataset, FourDGSdatasetDY
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_img_factor=1.0, smc_file=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        radius_extent = 1.0
        normalize_time = True
        self.is_dynerf = False
        self.is_hyper = False
        self.is_blender = False
        self.is_colmap = False
        self.is_dna = True if smc_file is not None else False

        if self.is_dna:
            scene_info = sceneLoadTypeCallbacks["DNA"](args.source_path, smc_file, factor=load_img_factor)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            self.is_colmap = True
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            normalize_time = False
            self.is_blender = True
        elif os.path.exists(os.path.join(args.source_path, "scene.json")):
            scene_info = sceneLoadTypeCallbacks["Hyper"](args.source_path, args.white_background, args.eval, gen_ply=True, factor=load_img_factor)
            # normalize_time = False
            self.is_hyper = True
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            radius_extent = 5.0
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
            self.is_dynerf = True
        else:
            print("error path: ", args.source_path)
            assert False, "Could not recognize scene type!"

        # if not self.loaded_iter:
        #     with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
        #         dest_file.write(src_file.read())
        #     json_cams = []
        #     camlist = []
        #     if scene_info.test_cameras:
        #         camlist.extend(scene_info.test_cameras)
        #     if scene_info.train_cameras:
        #         camlist.extend(scene_info.train_cameras)
        #     for id, cam in enumerate(camlist):
        #         json_cams.append(camera_to_JSON(id, cam))
        #     with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
        #         json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"] * radius_extent


        if self.is_hyper or self.is_blender:
            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
                self.gaussians.factor_t = True
        elif self.is_dynerf or self.is_dna:
            self.train_cameras = FourDGSdataset(scene_info.train_cameras, args)
            self.test_cameras = FourDGSdataset(scene_info.test_cameras, args)
            
        self.video_camera = None
        if scene_info.video_cameras is not None:
            self.video_camera = cameraList_from_camInfos(scene_info.video_cameras,-1,args)
        
        self.gaussians.max_frames = scene_info.maxtime
        self.gaussians.normalize_timestamp = normalize_time
        print(f"Max frames: {scene_info.maxtime}, normalize time: {normalize_time}")
        
        if self.is_dna or self.is_hyper:
            self.gaussians.factor_t = True

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(
                self.model_path,
                "point_cloud",
                "iteration_" + str(self.loaded_iter),
                "point_cloud.ply"
            ))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        if self.is_dynerf or self.is_dna:
            return self.train_cameras
        else:
            return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        if self.is_dynerf or self.is_dna:
            return self.test_cameras
        else:
            return self.test_cameras[scale]
    
    def getVideoCameras(self, scale=1.0):
        if self.is_dynerf or self.is_dna:
            return self.video_camera
        else:
            return self.video_camera[scale]
    
    
class DynamicScene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], only_frist=False, ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.is_dynerf = True

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        scene_info_list = []
        frame_id = args.source_path.split('/')[-1]
        model_root = args.model_path.replace('/'+frame_id, '/')
        num_frames = 300
        
        if only_frist:
            num_frames = 1
            
        cam_extrinsics = None
        cam_intrinsics = None
        for i in tqdm(range(0, num_frames), desc="Reading frames"):
            image_path = args.source_path.replace('/'+frame_id, f'/{str(i)}/input')
            scene_info, cam_extrinsics, cam_intrinsics = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, 
                image_path, 
                args.eval, 
                suppress=True, 
                timestamp=i, 
                gen_ply=True if i == 0 else False,
                cam_extrinsics=cam_extrinsics,
                cam_intrinsics=cam_intrinsics,
            )
            scene_info_list.append(scene_info)

        # import pdb; pdb.set_trace()

        scene_info = scene_info_list[0]
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            for scene_info in scene_info_list:
                random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info_list[0].nerf_normalization["radius"] * 5

        # for resolution_scale in resolution_scales:
        #     self.train_cameras[resolution_scale] = []
        #     for scene_info in tqdm(scene_info_list, desc="Loading training cameras"):
        #         self.train_cameras[resolution_scale].append(cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args))
                
        #     self.test_cameras[resolution_scale] = []
        #     for scene_info in tqdm(scene_info_list, desc="Loading Test Cameras"):
        #         self.test_cameras[resolution_scale].append(cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args))
        
        self.train_cameras = FourDGSdatasetDY(scene_info_list, args, split="train")
        self.test_cameras = FourDGSdatasetDY(scene_info_list, args, split="test")

        self.gaussians.max_frames = 300
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(
                self.model_path, 
                "point_cloud", 
                "iteration_" + str(self.loaded_iter),
                "point_cloud.ply"
            ))
        else:
            self.gaussians.create_from_pcd(scene_info_list[0].point_cloud, self.cameras_extent)
            

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0, deep_copy=False):
        # if deep_copy:
        #     return [cam_stack.copy() for cam_stack in self.train_cameras[scale]]
        return self.train_cameras

    def getTestCameras(self, scale=1.0, deep_copy=False):
        # if deep_copy:
        #     return [cam_stack.copy() for cam_stack in self.train_cameras[scale]]
        return self.test_cameras
    
    def getVideoCameras(self, scale=1.0):
        return self.video_camera