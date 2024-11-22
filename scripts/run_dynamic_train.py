import subprocess
from termcolor import colored

def safe_run(cmd):
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")

name = 'coffee_martini_dy_v2'
dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/coffee_martini/images_split'

# first frame
print(colored("Running: ", 'light_cyan'), f'frame: 0:')
command = [
    'python', 'train.py',
    '-s', f'{dataset_path}/0',
    '--model_path', f'output/{name}/0',
    '--iterations', '10000'
]
safe_run(command)

# second frame
print(colored("Running: ", 'light_cyan'), f'frame: 1:')
command = [
    'python', 'train.py',
    '-s', f'{dataset_path}/0',
    '--images', f'{dataset_path}/1/input',
    # '--densify_until_iter', '0',
    '--train_rest_frame',
    '--model_path', f'output/{name}/1',
    '--start_checkpoint', f'output/{name}/0/chkpnt10000.pth',
    '--iterations', '2000'
]
safe_run(command)
    
    
start = 2
for frame_id in range(start, 300):
    last_frame = frame_id - 1
    print(colored("Running: ", 'light_cyan'), f'frame: {frame_id}:')
    command = [
        'python', 'train.py',
        '-s', f'{dataset_path}/0',
        '--images', f'{dataset_path}/{frame_id}/input',
        # '--densify_until_iter', '0',
        '--train_rest_frame',
        # '--after_second_frame',
        '--model_path', f'output/{name}/{frame_id}',
        '--start_checkpoint', f'output/{name}/{last_frame}/chkpnt2000.pth',
        '--iterations', '2000'
    ]
    safe_run(command)