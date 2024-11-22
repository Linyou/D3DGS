import subprocess
from termcolor import colored
import os
from datetime import datetime
import time
from time import sleep

timestamp = time.time()
formatted_timestamp = datetime.fromtimestamp(timestamp).strftime('%Y%m%d-%H%M%S')

selected_gpu = '2'
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
    #     "path": "/data/loyot/datasets/cut-lemon1",
    #     "name": "cut-lemon1",
    # },
    # {
    #     "path": "/data/loyot/datasets/chickchicken",
    #     "name": "interp_chickchicken",
    # },
    # {
    #     "path": "/data/loyot/datasets/split-cookie",
    #     "name": "split-cookie",
    # },
    {
        "path": "/data/loyot/datasets/espresso",
        "name": "misc_espresso",
    },
    # {
    #     "path": "/data/loyot/datasets/americano",
    #     "name": "misc_americano",
    # },
    # {
    #     "path": "/data/loyot/datasets/vrig-3dprinter",
    #     "name": "vrig_3dprinter",
    # },
    # {
    #     "path": "/data/loyot/datasets/broom2",
    #     "name": "vrig_broom",
    # },
    # {
    #     "path": "/data/loyot/datasets/vrig-peel-banana",
    #     "name": "vrig_peel-banana",
    # },
    # {
    #     "path": "/data/loyot/datasets/vrig-chicken",
    #     "name": "vrig_chicken",
    # },
]


for task in hyper_list:
    name = task['name']
    print(colored("Running: ", 'light_cyan'), f'{name}')
    # for od in [8 , 16, 32, 64]:
    run_excu(task["name"], task["path"])
    sleep(5)

    