import concurrent.futures
import gc
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
import random
from torchvision import transforms as T
from utils.general_utils import PILtoTorch
from scene.SMCReader_cls import SMCReader
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.transform.camera.distortion import undistort_images

from xrprimer.transform.convention.camera import convert_camera_parameter

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    timestamp : float
    focal_length_x: float
    focal_length_y: float
    cx: float
    cy: float

class DNARendering(Dataset):
    def __init__(
        self,
        datadir,
        anno_smc,
        split="train",
        downsample=1.0,
    ):
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample
        self.anno_smc = SMCReader(anno_smc)

        self.load_meta()
        print(f"meta data loaded, total image:{len(self)}")

    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        Rs = []
        Ts = []
        img_paths = []
        instrinsics = []
        num_frames = len(self.anno_smc.smc['Mask']['0']['mask'])
        num_cams = 48
        mask = self.anno_smc.get_mask(0, 0)
        # self.img_hw = mask.shape
        self.img_wh = (
            int(mask.shape[1] * self.downsample),
            int(mask.shape[0] * self.downsample),
        )
        skip = 1 if self.split == "train" else 10
        
        cam_params = []
        for vid in range(num_cams):
            cam_param = self.anno_smc.get_Calibration(vid)
            cam_params.append(cam_param)
            # pose = self.anno_smc.get_Calibration(vid)['RT']
            # R = pose[:3, :3]
            # T = pose[:3, 3]
            # Rt = getWorld2View2(R, T)
            # instrinsic = self.anno_smc.get_Calibration(vid)['K']
            # instrinsic_scaled = instrinsic * self.downsample
            # # poses.append(pose)
            # Rs.append(Rt[:3, :3])
            # Ts.append(Rt[:3, 3])
            # instrinsics.append(instrinsic_scaled)
            cam_name = f'cam_{vid:02d}'
            img_cam_paths = []
            for f_id in range(0, num_frames, skip):
                img_name = f'{f_id}/images/{cam_name}.png'
                img_cam_paths.append(
                    os.path.join(self.root_dir, img_name)
                )
            img_paths.append(img_cam_paths)
            
        # self.poses = np.array(poses)
        # self.R = np.array(Rs)
        # self.T = np.array(Ts)   
        # self.instrinsics = np.array(instrinsics)
        self.cam_params = cam_params
        self.img_paths = img_paths

        
        self.cam_number = num_cams
        self.time_number = num_frames

    def __len__(self):
        return self.cam_number*self.time_number
    
    def __getitem__(self,index):
        
        if self.split == "train":
            cam_id = random.randint(0,self.cam_number-1)
            time_id = random.randint(0,self.time_number-1)
        else:
            cam_id = index // self.time_number
            time_id = index % self.time_number
            
        cam_param = self.cam_params[cam_id]
        image_path = self.img_paths[cam_id][time_id]
        img = Image.open(image_path)
        
        # Camera_id = str(cam_id)
        # camera_parameter = FisheyeCameraParameter(name=Camera_id)
        K = cam_param['K']
        # D = cam_param['D'] # k1, k2, p1, p2, k3
        RT = cam_param['RT']
        R = RT[:3, :3]
        T = RT[:3, 3]
        extrinsic = cam_param['RT']
        

        r_mat_inv = extrinsic[:3, :3]
        r_mat = np.linalg.inv(r_mat_inv)
        r_mat[0:3, 1:3] *= -1
        R = -np.transpose(r_mat)
        R[:,0] = -R[:,0]
        T = -T
        # T = c2ws[:3, 3]
        # T[2] = -T[2]
        # t_vec = extrinsic[:3, 3:]
        # t_vec = -np.dot(r_mat, t_vec).reshape((3))
        # R = r_mat
        # T = t_vec
        # R = -np.transpose(c2ws)
        # Rt = getWorld2View2(R, T)
        # R = c2ws[:3, :3]
        # T = c2ws[:3, 3]
        # Rt = getWorld2View2(R, T)
        # R = Rt[:3, :3]
        # T = Rt[:3, 3]
        # R[:,0] = -R[:,0]
        # R[:,2] = -R[:,2]
        # T = -matrix[:3, 3]
        # dist_coeff_k = [D[0],D[1],D[4]]
        # dist_coeff_p = D[2:4]
        # camera_parameter.set_KRT(K, R, T)
        # camera_parameter.set_dist_coeff(dist_coeff_k, dist_coeff_p)
        # camera_parameter.inverse_extrinsic()
        # # print("img: ", img.size)
        # # print("mask: ", self.img_wh)
        # camera_parameter.set_resolution(img.size[0], img.size[0])
        
        # corrected_cam, corrected_img = undistort_images(camera_parameter, np.array(img)[None])
        # # print("corrected_cam: ", corrected_cam)
        # # print("corrected_img: ", corrected_img)
        # K = np.asarray(corrected_cam.get_intrinsic())
        # R = np.asarray(corrected_cam.get_extrinsic_r())
        # T = np.asarray(corrected_cam.get_extrinsic_t())
        
        # print(R.shape)
        # corrected_img = Image.fromarray(corrected_img[0])
        K = K * self.downsample
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = PILtoTorch(img, None)
        img = img.to(torch.float32)
        # R = self.R[cam_id]
        # T = self.T[cam_id]
        FovX = focal2fov(K[0, 0], self.img_wh[0])
        FovY = focal2fov(K[1, 1], self.img_wh[1])
        # print(f"FovX: {FovX}, FovY: {FovY}")
        # print(f"focalx: {K[0, 0]}, focaly: {K[1, 1]}")
        
        image_name = f'{cam_id}_{time_id}'
        cx = K[0, 2]
        cy = K[1, 2]
        caminfo = CameraInfo(
            uid=cam_id, R=R, T=T, 
            FovY=FovY, FovX=FovX, 
            image=img,
            image_path=image_path, 
            image_name=image_name, 
            width=self.img_wh[0], height=self.img_wh[1], 
            timestamp=time_id,
            focal_length_x=K[0, 0],
            focal_length_y=K[1, 1],
            cx=cx,
            cy=cy,
        )
    
        
        return caminfo
    
    def load_pose(self,index):
        cam_param = self.cam_params[index]
        image_path = self.img_paths[index][0]
        # img = Image.open(image_path)
        
        # Camera_id = str(index)
        # camera_parameter = FisheyeCameraParameter(name=Camera_id)
        K = cam_param['K']
        D = cam_param['D'] # k1, k2, p1, p2, k3
        RT = cam_param['RT']
        R = RT[:3, :3]
        T = RT[:3, 3]
        extrinsic = cam_param['RT']
        r_mat_inv = extrinsic[:3, :3]
        r_mat = np.linalg.inv(r_mat_inv)
        t_vec = extrinsic[:3, 3:]
        t_vec = -np.dot(r_mat, t_vec).reshape((3))
        R = r_mat
        T = t_vec
        matrix = np.eye(4)
        matrix[:3,:3] = R
        matrix[:3, 3] = T
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        # dist_coeff_k = [D[0],D[1],D[4]]
        # dist_coeff_p = D[2:4]
        # camera_parameter.set_KRT(K, R, T)
        # camera_parameter.set_dist_coeff(dist_coeff_k, dist_coeff_p)
        # camera_parameter.inverse_extrinsic()
        # camera_parameter.set_resolution(img.size[0], img.size[0])
        
        # corrected_cam, corrected_img = undistort_images(camera_parameter, np.array(img)[None])
        # # print("corrected_cam: ", corrected_cam)
        # # print("corrected_img: ", corrected_img)
        # K = np.asarray(corrected_cam.get_intrinsic())
        # R = np.asarray(corrected_cam.get_extrinsic_r())
        # T = np.asarray(corrected_cam.get_extrinsic_t())
        
        # print(R.shape)
            
        K = K * self.downsample
        
        return R, T, K
