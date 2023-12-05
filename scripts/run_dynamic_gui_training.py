import os
import subprocess
from termcolor import colored

def safe_run(cmd):
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")

name = 'coffee_martini_poly_base_gui'
if os.path.exists(f'output/{name}/'):
    print(colored("Warning: ", 'yellow'), f'output/{name}/ already exists, removing')
    # remove all file in output/name
    os.system(f'rm -rf output/{name}')
    
    
dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/coffee_martini/images_split'

command = [
    'python', 'gui_training.py',
    '-s', f'{dataset_path}/0',
    '--dynamic',
    # '-r', '1',
    '--model_path', f'output/{name}/',
    '--eval'
]
safe_run(command)

