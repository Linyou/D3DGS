import subprocess
from termcolor import colored
import os
from datetime import datetime
import time
from time import sleep

timestamp = time.time()
formatted_timestamp = datetime.fromtimestamp(timestamp).strftime('%Y%m%d-%H%M%S')

selected_gpu = '0'
my_env = os.environ.copy()
my_env["CUDA_VISIBLE_DEVICES"] = selected_gpu

def safe_run(cmd):
    # Run the command
    try:
        subprocess.run(cmd, env=my_env, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")


def run_excu(name_prefix, path, order=32):
    tag = formatted_timestamp
    name = f'hypernerf/{name_prefix}_fftpoly@{tag}'
    # name = f'{name_prefix}_fftpoly_60k'
    # name = f'{name_prefix}_fftpoly_order{order}'
    dataset_path = path
    config = "arguments/hypernerf/vrig.py"

    # first frame
    # print(colored("Running: ", 'light_cyan'), f'frame: 0:')
    command = [
        'python', 'train.py',
        '-s', f'{dataset_path}',
        '--model_path', f'output/{name}/',
        # '--iterations', '30000',
        '--config', f'{config}',
        # '--test_iterations', '59999',
        '--test_iterations', '2000',
        '--eval',
        # '--xyz_traj_feat_dim', f'{order}',
    ]
    safe_run(command)
    
hyper_list = [
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/interp_cut-lemon/cut-lemon1",
    #     "name": "interp_cut-lemon",
    # },
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/interp_chickchicken/chickchicken",
    #     "name": "interp_chickchicken",
    # },
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/misc_split-cookie/split-cookie",
    #     "name": "misc_split-cookie",
    # },
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/misc_espresso/espresso",
    #     "name": "misc_espresso",
    # },
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/misc_americano/americano",
    #     "name": "misc_americano",
    # },
    {
        "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/vrig_3dprinter/3dprinter",
        "name": "vrig_3dprinter",
    },
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/vrig_broom/broom",
    #     "name": "vrig_broom",
    # },
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/vrig_peel-banana/peel-banana",
    #     "name": "vrig_peel-banana",
    # },
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/vrig_chicken/chicken",
    #     "name": "vrig_chicken",
    # },
]


for task in hyper_list:
    name = task['name']
    print(colored("Running: ", 'light_cyan'), f'{name}')
    # for od in [8 , 16, 32, 64]:
    run_excu(task["name"], task["path"])
    sleep(5)

    