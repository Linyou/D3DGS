import subprocess
from termcolor import colored
import os
import time
from datetime import datetime

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


def run_excu(name_prefix, path):
    tag = formatted_timestamp
    name = f'dnerf/{name_prefix}_fftpoly@{tag}'
    dataset_path = path
    config = f"arguments/dnerf/{name_prefix}.py"

    # first frame
    print(colored("Running: ", 'light_cyan'), f'frame: 0:')
    command = [
        'python', 'train.py',
        '-s', f'{dataset_path}',
        '--model_path', f'output/{name}/',
        # '--iterations', '30000',
        '--config', f'{config}',
        '--test_iterations', '2000',
        '--eval'
    ]
    safe_run(command)
    
hyper_list = [
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/dynamic_data/lego",
    #     "name": "lego",
    # },
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/dynamic_data/bouncingballs",
    #     "name": "bouncingballs",
    # },
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/dynamic_data/hellwarrior",
    #     "name": "hellwarrior",
    # },
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/dynamic_data/hook",
    #     "name": "hook",
    # },
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/dynamic_data/jumpingjacks",
    #     "name": "jumpingjacks",
    # },
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/dynamic_data/mutant",
    #     "name": "mutant",
    # },
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/dynamic_data/standup",
    #     "name": "standup",
    # },
    {
        "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/dynamic_data/trex",
        "name": "trex",
    },
]

for task in hyper_list:
    print(f"Running {task['name']}")
    run_excu(task["name"], task["path"])

    