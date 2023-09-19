import subprocess
from termcolor import colored

def safe_run(cmd):
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")

name = 'coffee_martini_poly_base_v5'
dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/coffee_martini/images_split'

# first frame
print(colored("Running: ", 'light_cyan'), f'frame: 0:')
command = [
    'python', 'train.py',
    '-s', f'{dataset_path}/0',
    '--model_path', f'output/{name}/',
    '--iterations', '180000',
]
safe_run(command)
    