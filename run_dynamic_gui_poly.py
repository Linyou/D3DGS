import subprocess
from termcolor import colored

def safe_run(cmd):
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")

dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/coffee_martini/images_split'

command = [
    'python', 'gui.py',
    '-s', f'{dataset_path}/0',
    '--dynamic', 
    '--model_path', f'output/coffee_martini_poly_base_v4',
    '--start_checkpoint', f'output/coffee_martini_poly_base_v4/chkpnt180000.pth',
]
safe_run(command)