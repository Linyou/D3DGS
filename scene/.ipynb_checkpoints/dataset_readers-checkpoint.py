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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import copy
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.hyper_loader import Load_hyper_data,format_hyper_data
from scene.dnerf_loader import read_timeline, generateCamerasFromTransforms
from utils.general_utils import PILtoTorch
from tqdm import tqdm
import torchvision.transforms as transforms

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
    timestamp: int

class CameraInfoExtra(NamedTuple):
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
    timestamp: int
    focal_length: float
    cx: float
    cy: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    train_cameras_0: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, suppress=False, timestamp=0):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        if not suppress:
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
            sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        # print(intr)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)
        image = image_path

        cam_info = CameraInfoExtra(
            uid=uid, R=R, T=T, 
            FovY=FovY, FovX=FovX, image=image,
            image_path=image_path, image_name=image_name, 
            width=width, height=height, timestamp=timestamp,
            focal_length=focal_length_x,
            cx=cx, cy=cy,
        )
        cam_infos.append(cam_info)
    if not suppress:
        sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, suppress=False, dynamic=False, timestamp=0, gen_ply=True, cam_extrinsics=None, cam_intrinsics=None):
    
    if cam_extrinsics is None or cam_intrinsics is None:
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
            
    # import pdb; pdb.set_trace()

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, 
        cam_intrinsics=cam_intrinsics, 
        images_folder=os.path.join(path, reading_dir), 
        suppress=True, 
        timestamp=timestamp
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # import pdb; pdb.set_trace()
    # if eval:
    #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    # else:
    train_cam_infos = cam_infos[1:]
    test_cam_infos = cam_infos[0:1]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
            
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
        
    if gen_ply:
        # Since this data set has no colmap data, we start with random points
        num_pts = 50_000
        print(f"Generating random point cloud ({num_pts})...")
        radius = nerf_normalization["radius"] * 5
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2 * radius - radius
        shs = np.random.random((num_pts, 3)) / 255.0
        cat_xyz = np.concatenate((pcd.points, xyz), axis=0)
        cat_shs = np.concatenate((pcd.colors, SH2RGB(shs)), axis=0)
        pcd = BasicPointCloud(points=cat_xyz, colors=cat_shs, normals=np.zeros((num_pts, 3)))

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        train_cameras_0=None,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        maxtime=None,
        video_cameras=None,
    )
    return scene_info, cam_extrinsics, cam_intrinsics

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            time = mapper[frame["time"]]
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = PILtoTorch(image,(800,800))
            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[1], height=image.shape[2],
                            timestamp = time))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    timestamp_mapper, max_time = read_timeline(path)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, timestamp_mapper)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, timestamp_mapper)
    video_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json", extension, max_time)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # Since this data set has no colmap data, we start with random points
    num_pts = 10_000
    print(f"Generating random point cloud ({num_pts})...")
    
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        train_cameras_0=None,
        test_cameras=test_cam_infos,
        video_cameras=video_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        maxtime=int(max_time),
    )
    return scene_info


def readHyperDataInfos(datadir,use_bg_points,eval, gen_ply=False, factor=1.0):
    # train_cam_infos = Load_hyper_data(datadir,1.0,use_bg_points,split ="train")
    # test_cam_infos = Load_hyper_data(datadir,1.0,use_bg_points,split="test")
    print("Loading hyper data with factor: ", factor)
    
    train_cam_infos = Load_hyper_data(datadir,factor,use_bg_points,split ="train")
    test_cam_infos = Load_hyper_data(datadir,factor,use_bg_points,split="test")

    train_cam = format_hyper_data(train_cam_infos,"train")
    max_time = train_cam_infos.max_time
    video_cam_infos = copy.deepcopy(test_cam_infos)
    video_cam_infos.split="video"

    ply_path = os.path.join(datadir, "points.npy")

    xyz = np.load(ply_path,allow_pickle=True)
    xyz -= train_cam_infos.scene_center
    xyz *= train_cam_infos.coord_scale
    xyz = xyz.astype(np.float32)
    shs = np.random.random((xyz.shape[0], 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))

    nerf_normalization = getNerfppNorm(train_cam)

    if gen_ply:
        # Since this data set has no colmap data, we start with random points
        num_pts = 50_000
        print(f"Generating random point cloud ({num_pts})...")
        radius = nerf_normalization["radius"] * 5
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2 * radius - radius
        shs = np.random.random((num_pts, 3)) / 255.0
        cat_xyz = np.concatenate((pcd.points, xyz), axis=0)
        cat_shs = np.concatenate((pcd.colors, SH2RGB(shs)), axis=0)
        pcd = BasicPointCloud(points=cat_xyz, colors=cat_shs, normals=np.zeros((num_pts, 3)))

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        train_cameras_0=None,
        test_cameras=test_cam_infos,
        video_cameras=video_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        maxtime=max_time
    )
    return scene_info

def format_infos_dna(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    
    for idx in tqdm(range(dataset.cam_number)):
        image_path = None
        image_name = f"{idx}"
        # matrix = np.linalg.inv(np.array(pose))
        R, T, K= dataset.load_pose(idx)
        FovX = K[0, 0]
        FovY = K[1, 1]
        cameras.append(CameraInfo(
            uid=idx, R=R, T=T, 
            FovY=FovY, FovX=FovX, 
            image=image,
            image_path=image_path, 
            image_name=image_name, 
            width=dataset.img_wh[0], 
            height=dataset.img_wh[1],
            timestamp=0
        ))

    return cameras

def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                timestamp = time))

    return cameras

def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(
            uid=idx, R=R, T=T, 
            FovY=FovY, FovX=FovX, 
            image=image,
            image_path=image_path, 
            image_name=image_name, 
            width=image.shape[2], 
            height=image.shape[1],
            timestamp=time
        ))
    return cameras

def readdynerfInfo(datadir,use_bg_points,eval):
    # loading all the data follow hexplane format
    ply_path = os.path.join(datadir, "points3d.ply")

    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
        datadir,
        "train",
        1.0,
        time_scale=1,
        scene_bbox_min=[-2.5, -2.0, -1.0],
        scene_bbox_max=[2.5, 2.0, 1.0],
        eval_index=0,
    )    
    test_dataset = Neural3D_NDC_Dataset(
        datadir,
        "test",
        1.0,
        time_scale=1,
        scene_bbox_min=[-2.5, -2.0, -1.0],
        scene_bbox_max=[2.5, 2.0, 1.0],
        eval_index=0,
    )
    train_cam_infos = format_infos(train_dataset, "train")
    
    # test_cam_infos = format_infos(test_dataset,"test")
    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)
    # create pcd
    # if not os.path.exists(ply_path):
    # Since this data set has no colmap data, we start with random points
    num_pts = 50_000
    print(f"Generating random point cloud ({num_pts})...")
    radius = nerf_normalization["radius"] * 5
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.random((num_pts, 3)) * 2 * radius - radius
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        # xyz = np.load
        pcd = fetchPly(ply_path)
    except:
        pcd = None
        
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_dataset,
        train_cameras_0=None,
        test_cameras=test_dataset,
        video_cameras=val_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        maxtime=300
    )
    return scene_info

def readdnaInfo(datadir, anno_smc, factor=1.0):
    # loading all the data follow hexplane format
    ply_path = os.path.join(datadir, "points3d.ply")

    from scene.dna_rendering import DNARendering
    train_dataset = DNARendering(
        datadir,
        anno_smc,
        "train",
        factor,
    )   
    test_dataset = DNARendering(
        datadir,
        anno_smc,
        "test",
        factor,
    )  
    train_dataset_0 = DNARendering(
        datadir,
        anno_smc,
        "train",
        factor,
        only_0=True,
    )     
    # train_cam_infos = format_infos_dna(train_dataset, "train")
    # nerf_normalization = getNerfppNorm(train_cam_infos)
    # create pcd
    # if not os.path.exists(ply_path):
    # Since this data set has no colmap data, we start with random points
    num_pts = 10000
    radius = 3.0
    print(f"Generating random point cloud ({num_pts})...")
    print("radius: ", radius)
    nerf_normalization = {"radius": radius}
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.random((num_pts, 3)) * 2 * radius - radius
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # try:
    #     # xyz = np.load
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None
        
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_dataset,
        test_cameras=test_dataset,
        train_cameras_0=train_dataset_0,
        video_cameras=None,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        maxtime=train_dataset.time_number,
    )
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Hyper": readHyperDataInfos,
    "dynerf" : readdynerfInfo,
    "DNA" : readdnaInfo,
}