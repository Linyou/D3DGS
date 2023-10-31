import subprocess
from termcolor import colored

def safe_run(cmd):
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")

# name = 'flame_steak_poly_base_v10'
# dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/flame_steak/images_split/0'

name = 'lego_poly_base_v12'
dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/dynamic_data/lego'
config = "arguments/dnerf/lego.py"

# name = 'vrig_chicken_poly_base_v13'
# dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/vrig_chicken/chicken'

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
    