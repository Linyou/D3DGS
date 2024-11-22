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


def run_excu(name_prefix, path, smc):
    tag = formatted_timestamp
    name = f'dna/{name_prefix}_fftpoly@{tag}'
    dataset_path = path
    config = "arguments/dna/default.py"

    # first frame
    print(colored("Running: ", 'light_cyan'), f'frame: 0:')
    command = [
        'python', 'train.py',
        '-s', f'{dataset_path}',
        '--model_path', f'output/{name}/',
        # '--iterations', '30000',
        '--config', f'{config}',
        '--smc_file', f'{smc}',
        '--test_iterations', '2000',
        '--eval'
    ]
    safe_run(command)
    
hyper_list = [
    {
        "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/dna_rendering/0012_09",
        "smc": f"/home/loyot/Downloads/0012_09_annots.smc",
        "name": "0012_09",
    },
    # {
    #     "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/dna_rendering/0008_01",
    #     "smc": f"/home/loyot/Downloads/0008_01_annots.smc",
    #     "name": "0008_01",
    # },
]

for task in hyper_list:
    print(f"Running {task['name']}")
    run_excu(task["name"], task["path"], task["smc"])

    