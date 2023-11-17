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
    name = f'{name_prefix}_fftpoly_60k'
    dataset_path = path
    config = "arguments/hypernerf/default.py"

    # first frame
    print(colored("Running: ", 'light_cyan'), f'frame: 0:')
    command = [
        'python', 'train.py',
        '-s', f'{dataset_path}',
        '--model_path', f'output/{name}/',
        # '--iterations', '30000',
        '--config', f'{config}',
        '--test_iterations', '59999',
        # '--test_iterations', '29999',
        '--eval'
    ]
    safe_run(command)
    
hyper_list = [
    {
        "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/vrig_3dprinter/3dprinter",
        "name": "vrig_3dprinter",
    },
    {
        "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/vrig_broom/broom",
        "name": "vrig_broom",
    },
    {
        "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/vrig_peel-banana/peel-banana",
        "name": "vrig_peel-banana",
    },
    {
        "path": "/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/vrig_chicken/chicken",
        "name": "vrig_chicken",
    },
]

for task in hyper_list:
    print(f"Running {task['name']}")
    run_excu(task["name"], task["path"])

    