import warnings

warnings.filterwarnings("ignore")

import json
import os
import random

import cv2

import numpy as np
import torch
from PIL import Image
import math
from tqdm import tqdm
from scene.utils import Camera
from typing import NamedTuple
from torch.utils.data import Dataset
from utils.general_utils import PILtoTorch
# from scene.dataset_readers import 
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import copy
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


class Load_hyper_data(Dataset):
    def __init__(self, 
                 datadir, 
                 ratio=1.0,
                 use_bg_points=False,
                 split="train"
                 ):
        
        from .utils import Camera
        datadir = os.path.expanduser(datadir)
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)

        self.near = scene_json['near']
        self.far = scene_json['far']
        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']

        self.all_img = dataset_json['ids']
        self.val_id = dataset_json['val_ids']
        self.split = split
        if len(self.val_id) == 0:
            self.i_train = np.array([i for i in np.arange(len(self.all_img)) if
                            (i%4 == 0)])
            self.i_test = self.i_train+2
            self.i_test = self.i_test[:-1,]
        else:
            self.train_id = dataset_json['train_ids']
            self.i_test = []
            self.i_train = []
            for i in range(len(self.all_img)):
                id = self.all_img[i]
                if id in self.val_id:
                    self.i_test.append(i)
                if id in self.train_id:
                    self.i_train.append(i)
        

        self.all_cam = [meta_json[i]['camera_id'] for i in self.all_img]
        self.all_time = [meta_json[i]['warp_id'] for i in self.all_img]
        max_time = max(self.all_time)
        # self.all_time = [meta_json[i]['warp_id']/max_time for i in self.all_img]
        
        # import pdb; pdb.set_trace()
        self.selected_time = set(self.all_time)
        self.ratio = ratio
        self.max_time = max(self.all_time)
        self.min_time = min(self.all_time)
        self.i_video = [i for i in range(len(self.all_img))]
        self.i_video.sort()
        # all poses
        self.all_cam_params = []
        for im in self.all_img:
            camera = Camera.from_json(f'{datadir}/camera/{im}.json')
            camera = camera.scale(ratio)
            camera.position -=  self.scene_center
            camera.position *=  self.coord_scale
            self.all_cam_params.append(camera)

        self.all_img = [f'{datadir}/rgb/{int(1/ratio)}x/{i}.png' for i in self.all_img]
        self.h, self.w = self.all_cam_params[0].image_shape
        self.map = {}
        self.image_one = Image.open(self.all_img[0])
        self.image_one_torch = PILtoTorch(self.image_one,None).to(torch.float32)
        
    def __getitem__(self, index):
        if self.split == "train":
            # index = random.randint(0, len(self.i_train))
            return self.load_raw(index)
 
        elif self.split == "test":
            return self.load_raw(self.i_test[index])
        elif self.split == "video":
            return self.load_video(self.i_video[index])
    def __len__(self):
        if self.split == "train":
            return len(self.i_train)
        elif self.split == "test":
            return len(self.i_test)
        elif self.split == "video":
            # return len(self.i_video)
            return len(self.video_v2)
    def load_video(self, idx):
        if idx in self.map.keys():
            return self.map[idx]
        camera = self.all_cam_params[idx]
        w = self.image_one.size[0]
        h = self.image_one.size[1]
        # image = PILtoTorch(image,None)
        # image = image.to(torch.float32)
        time = self.all_time[idx]
        R = camera.orientation.T
        T = - camera.position @ R
        FovY = focal2fov(camera.focal_length, self.h)
        FovX = focal2fov(camera.focal_length, self.w)
        image_path = "/".join(self.all_img[idx].split("/")[:-1])
        image_name = self.all_img[idx].split("/")[-1]
        caminfo = CameraInfo(
            uid=idx, R=R, T=T, 
            FovY=FovY, FovX=FovX, 
            image=self.image_one_torch,
            image_path=image_path, 
            image_name=image_name, 
            width=w, height=h, 
            timestamp=time,
            focal_length_x=camera.focal_length,
            focal_length_y=camera.focal_length,
            cx=camera.principal_point_x,
            cy=camera.principal_point_y,
        )
        self.map[idx] = caminfo
        return caminfo  
    def load_raw(self, idx):
        if idx in self.map.keys():
            return self.map[idx]
        camera = self.all_cam_params[idx]
        image = Image.open(self.all_img[idx])
        
        # undisort the image
        # image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        
        # K = np.zeros((3, 3))
        # K[0, 0] = camera.focal_length
        # K[1, 1] = camera.focal_length
        # K[0, 2] = camera.principal_point_x
        # K[1, 2] = camera.principal_point_y
        # K[2, 2] = 1

        # distCoeffs = np.float32([
        #     camera.radial_distortion[0],
        #     camera.radial_distortion[1],
        #     camera.tangential_distortion[0],
        #     camera.tangential_distortion[1],
        #     camera.radial_distortion[2],
        # ])
        # print("start undistort")
        # new_k = np.zeros((3, 3))
        # new_k, _ = cv2.getOptimalNewCameraMatrix(
        #     K, distCoeffs, (w, h), 1, (w, h)
        # )
        # img_undistored = cv2.undistort(image, K, distCoeffs)
        # image = Image.fromarray(cv2.cvtColor(img_undistored, cv2.COLOR_BGR2RGB)) 
        # print("K: ", K)
        # print("new_k: ", new_k)
        # print("end undistort")
        
        w = image.size[0]
        h = image.size[1]
        image = PILtoTorch(image,None)
        image = image.to(torch.float32)
        time = self.all_time[idx]
        R = camera.orientation.T
        T = - camera.position @ R
        FovY = focal2fov(camera.focal_length, self.h)
        FovX = focal2fov(camera.focal_length, self.w)
        image_path = "/".join(self.all_img[idx].split("/")[:-1])
        image_name = self.all_img[idx].split("/")[-1]
        caminfo = CameraInfo(
            uid=idx, R=R, T=T, 
            FovY=FovY, FovX=FovX, 
            image=image,
            image_path=image_path, 
            image_name=image_name, 
            width=w, height=h, 
            timestamp=time,
            focal_length_x=camera.focal_length,
            focal_length_y=camera.focal_length,
            cx=camera.principal_point_x,
            cy=camera.principal_point_y,
        )
        self.map[idx] = caminfo
        return caminfo  

        
def format_hyper_data(data_class, split):
    if split == "train":
        data_idx = data_class.i_train
    elif split == "test":
        data_idx = data_class.i_test
    # dataset = data_class.copy()
    # dataset.mode = split
    cam_infos = []
    for uid, index in tqdm(enumerate(data_idx)):
        camera = data_class.all_cam_params[index]
        # image = Image.open(data_class.all_img[index])
        # image = PILtoTorch(image,None)
        time = data_class.all_time[index]
        R = camera.orientation.T
        T = - camera.position @ R
        FovY = focal2fov(camera.focal_length, data_class.h)
        FovX = focal2fov(camera.focal_length, data_class.w)
        image_path = "/".join(data_class.all_img[index].split("/")[:-1])
        image_name = data_class.all_img[index].split("/")[-1]
        cam_info = CameraInfo(
            uid=uid, R=R, T=T, 
            FovY=FovY, FovX=FovX, 
            image=None,
            image_path=image_path, 
            image_name=image_name, 
            width=int(data_class.w), 
            height=int(data_class.h), 
            timestamp=time,
            focal_length_x=camera.focal_length,
            focal_length_y=camera.focal_length,
            cx=camera.principal_point_x,
            cy=camera.principal_point_y,
        )
        cam_infos.append(cam_info)
    return cam_infos
        # matrix = np.linalg.inv(np.array(poses))
        # R = -np.transpose(matrix[:3,:3])
        # R[:,0] = -R[:,0]
        # T = -matrix[:3, 3]