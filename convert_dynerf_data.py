import os
from glob import glob
import subprocess
from termcolor import colored
from tqdm import tqdm

def safe_run(cmd):
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")

name_list = ["cook_spinach", "cut_roasted_beef", "flame_steak", "sear_steak"]
# "flame_salmon_1"

for name in name_list:
    root = f"/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/{name}/images_split/0/"
    frames = glob(root+"*")
    frames = [f.split("/")[-1] for f in frames]
    print(colored("Running: ", 'light_cyan'), f'Data: {name}, Frames: {len(frames)}')
    # for f_id in tqdm(frames):
    #     mkdir_cmd = f"mkdir /home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/{name}/images_split/{f_id}/input"
    #     mv_cmd = f"mv /home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/{name}/images_split/{f_id}/*.png /home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/{name}/images_split/{f_id}/input/"
        
    #     # run command
    #     os.system(mkdir_cmd)
    #     os.system(mv_cmd)
    # run convert command
    
    convert_cmd = f'python convert.py -s {root}'
    os.system(convert_cmd)
        