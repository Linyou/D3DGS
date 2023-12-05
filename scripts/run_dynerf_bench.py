import subprocess
from termcolor import colored
import os


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


def run_excu(name_prefix, path):
    name = f'dynerf_new/{name_prefix}_fftpoly_60K_v2'
    # name = f'{name_prefix}_fftpoly'
    dataset_path = path
    config = "arguments/dynerf/default.py"

    # first frame
    print(colored("Running: ", 'light_cyan'), f'frame: 0:')
    command = [
        'python', 'train.py',
        '-s', f'{dataset_path}',
        '--model_path', f'output/{name}/',
        # '--iterations', '30000',
        '--config', f'{config}',
        # '--test_iterations', '59999',
        '--test_iterations', '10000',
        '--eval'
    ]
    safe_run(command)
    
hyper_list = [
    {
        "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/flame_steak/images_split/0",
        "name": "flame_steak",
    },
    {
        "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/cut_roasted_beef/images_split/0",
        "name": "cut_roasted_beef",
    },
    {
        "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/cook_spinach/images_split/0",
        "name": "cook_spinach",
    },
    {
        "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/sear_steak/images_split/0",
        "name": "sear_steak",
    },
    {
        "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/coffee_martini/images_split/0",
        "name": "coffee_martini",
    },
    {
        "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/flame_salmon_1/images_split/0",
        "name": "flame_salmon_1",
    },
]

for task in hyper_list:
    print(f"Running {task['name']}")
    run_excu(task["name"], task["path"])

    