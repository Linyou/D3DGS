from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
import random
from PIL import Image
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        split="train",
    ):
        self.dataset = dataset
        self.args = args
        self.split = split
    def __getitem__(self, index):
        caminfo = None
        try:
            image, w2c, time = self.dataset[index]
            R,T = w2c
            FovX = focal2fov(self.dataset.focal[0], image.shape[2])
            FovY = focal2fov(self.dataset.focal[0], image.shape[1])
        except:
            caminfo = self.dataset[index]
            image = caminfo.image
            R = caminfo.R
            T = caminfo.T
            FovX = caminfo.FovX
            FovY = caminfo.FovY
            time = caminfo.timestamp

        return Camera(
            colmap_id=index, 
            R=R, T=T, 
            FoVx=FovX, FoVy=FovY, 
            image=image, gt_alpha_mask=None,
            image_name=caminfo.image_name, uid=index, 
            data_device=torch.device("cuda"),
            timestamp=time,
            extra_cam_info=caminfo if self.args.use_extra_cam_info else None,
        )
    def __len__(self):
        
        return len(self.dataset)
    
class FourDGSdatasetDY(Dataset):
    def __init__(
        self,
        dataset,
        args,
        split="train",
    ):
        self.dataset = dataset
        self.args = args
        self.split = split
    def __getitem__(self, index):
        
        

        if self.split == "train":
            selected_time = random.randint(0, len(self.dataset)-1)
            selec_scene_infos = self.dataset[selected_time]
            selec_cams = selec_scene_infos.train_cameras
            random_cam_index = random.randint(0, len(selec_cams)-1)
            caminfo = selec_cams[random_cam_index]
        elif self.split == "test":
            caminfo = self.dataset[index].test_cameras[0]
        
        image = caminfo.image
        R = caminfo.R
        T = caminfo.T
        FovX = caminfo.FovX
        FovY = caminfo.FovY
        time = caminfo.timestamp
        image = Image.open(caminfo.image)
        orig_w, orig_h = image.size
        resolution = (int(orig_w), int(orig_h))
        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]

        return Camera(
            colmap_id=index, 
            R=R, T=T, 
            FoVx=FovX, FoVy=FovY, 
            image=gt_image, gt_alpha_mask=None,
            image_name=caminfo.image_name, uid=index, 
            data_device=torch.device("cuda"),
            timestamp=time,
            extra_cam_info=caminfo if self.args.use_extra_cam_info else None,
        )
    def __len__(self):
        if self.split == 'train':
            return len(self.dataset) * len(self.dataset[0].train_cameras)
        else :
            return len(self.dataset)
