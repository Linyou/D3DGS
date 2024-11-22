import subprocess
from termcolor import colored
import os
import time
from datetime import datetime

data_root = "/home/loyot/loyot/datasets/dynamic_nerf/"

timestamp = time.time()
formatted_timestamp = datetime.fromtimestamp(timestamp).strftime('%Y%m%d-%H%M%S')

selected_gpu = '7'
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
    
dnerf_list = [
    # {
    #     "path": os.path.join(data_root, "lego"),
    #     "name": "lego",
    # },
    # {
    #     "path": os.path.join(data_root, "bouncingballs"),
    #     "name": "bouncingballs",
    # },
    {
        "path": os.path.join(data_root, "hellwarrior"),
        "name": "hellwarrior",
    },
    # {
    #     "path": os.path.join(data_root, "hook"),
    #     "name": "hook",
    # },
    # {
    #     "path": os.path.join(data_root, "jumpingjacks"),
    #     "name": "jumpingjacks",
    # },
    # {
    #     "path": os.path.join(data_root, "mutant"),
    #     "name": "mutant",
    # },
    # {
    #     "path": os.path.join(data_root, "standup"),
    #     "name": "standup",
    # },
    # {
    #     "path": os.path.join(data_root, "trex"),
    #     "name": "trex",
    # },
]

for task in dnerf_list:
    print(f"Running {task['name']}")
    run_excu(task["name"], task["path"])

    