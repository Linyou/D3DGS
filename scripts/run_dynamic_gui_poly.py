import subprocess
from termcolor import colored

def safe_run(cmd):
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")

# scene
# ['vrig_broom','vrig_peel-banana', 'vrig_3dprinter', 'vrig_chicken']

name = "split-cookie"
ckpt_file = "chkpnt10000.pth"
# dataset_path = f'/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/{name}/images_split'
dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/misc_split-cookie/split-cookie'
output_name = f"hypernerf/misc_split-cookie_fftpoly@20231212-024358"
config = "arguments/hypernerf/default.py"
command = [
    'python', 'gui.py',
    # '-s', f'{dataset_path}/0',
    '--dynamic', 
    '-s', f'{dataset_path}',
    '--model_path', f'output/{output_name}',
    '--start_checkpoint', f'/home/loyot/workspace/code/gaussian-splatting/output/{output_name}/{ckpt_file}',
    '--configs', f'{config}',
]
safe_run(command) 