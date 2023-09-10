import subprocess
from termcolor import colored

def safe_run(cmd):
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")

dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/yundong'

# first frame
print(colored("Running: ", 'light_cyan'), f'frame: 0:')
command = [
    'python', 'gui_training.py',
    '-s', f'{dataset_path}/',
    # '-r', '1',
    '--model_path', 'output/yundong/',
]
safe_run(command)

