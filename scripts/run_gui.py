import subprocess
from termcolor import colored

def safe_run(cmd):
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")

name = 'coffee_martini_dy_v1'
dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/coffee_martini/images_split'
frame_id = 0
last_frame = 1
command = [
    'python', 'gui.py',
    '-s', f'{dataset_path}/0',
    '--images', f'{dataset_path}/{frame_id}/input',
    '--densify_until_iter', '0',
    '--model_path', f'output/{name}/{frame_id}',
    '--start_checkpoint', f'output/{name}/{last_frame}/chkpnt2000.pth',
    '--iterations', '2000'
]
safe_run(command)