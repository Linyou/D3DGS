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
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, getWorld2View

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
        only_0=False,
    ):
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample
        self.anno_smc = SMCReader(anno_smc)
        self.transform = T.ToTensor()
        

        self.load_meta(only_0)
        print(f"meta data loaded, total image:{len(self)}")

    def load_meta(self, only_0=False):
        """
        Load meta data from the dataset.
        """
        Rs = []
        Ts = []
        img_paths = []
        instrinsics = []
        if only_0:
            num_frames = 1
        else:
            num_frames = len(self.anno_smc.smc['Mask']['0']['mask'])
            num_frames = int(num_frames/2)
            
        num_cams = 48
        mask = self.anno_smc.get_mask(0, 0)
        # self.img_hw = mask.shape
        self.img_wh = (
            int(mask.shape[1] * self.downsample),
            int(mask.shape[0] * self.downsample),
        )
        
        cam_params = []
        self.timestamps = []
        all_images = []
        all_images_time = []
        all_cameras = []
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
            for f_id in range(0, num_frames):
                if vid == 0:
                    self.timestamps.append(f_id)
                img_name = f'{f_id}/images/{cam_name}.png'
                img_cam_paths.append(
                    os.path.join(self.root_dir, img_name)
                )
                all_images.append(os.path.join(self.root_dir, img_name))
                all_images_time.append(f_id)
                all_cameras.append({
                    "id": vid,
                    "cam_param": cam_param,
                })
            img_paths.append(img_cam_paths)
            
        # self.poses = np.array(poses)
        # self.R = np.array(Rs)
        # self.T = np.array(Ts)   
        # self.instrinsics = np.array(instrinsics)
        self.cam_params = cam_params
        self.img_paths = img_paths
        self.all_images = all_images
        self.all_images_time = all_images_time
        self.all_cameras = all_cameras

        
        self.cam_number = num_cams
        self.time_number = num_frames

    def __len__(self):
        if self.split == "train":
            return len(self.all_images)
        elif self.split == "test":
            return self.cam_number
    
    def __getitem__(self, index):
        
        if self.split == "train":
            cam_id = self.all_cameras[index]['id']
            time_id = self.all_images_time[index]
            cam_param = self.all_cameras[index]['cam_param']
            image_path = self.all_images[index]
        else:
            cam_id = index 
            time_id = int(index * 3 % self.time_number)
            cam_param = self.cam_params[cam_id]
            image_path = self.img_paths[cam_id][time_id]
            
        img = Image.open(image_path)
        
        # Camera_id = str(cam_id)
        # camera_parameter = FisheyeCameraParameter(name=Camera_id)
        K = cam_param['K']
        c2w = cam_param['RT']
        

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        
        # print(R.shape)
        # corrected_img = Image.fromarray(corrected_img[0])
        K = K * self.downsample
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)
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
    
    def load_pose(self, index):
        cam_param = self.cam_params[index]
        K = cam_param['K']
        c2w = cam_param['RT']
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        K = K * self.downsample
        FovX = focal2fov(K[0, 0], self.img_wh[0])
        FovY = focal2fov(K[1, 1], self.img_wh[1])
        return R, T, FovX, FovY
    
