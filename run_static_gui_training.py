import subprocess
from termcolor import colored

def safe_run(cmd):
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")

# dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/yundong'
# dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/360_v2/garden'
dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/nerf_synthetic/lego'

command = [
    'python', 'gui_training.py',
    '-s', f'{dataset_path}/',
    # '-r', '1',
    '--model_path', 'output/temp/',
]
safe_run(command)

