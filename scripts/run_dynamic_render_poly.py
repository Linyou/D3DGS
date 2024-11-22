import subprocess
from termcolor import colored
import os
from time import sleep

selected_gpu = '6'
my_env = os.environ.copy()
my_env["CUDA_VISIBLE_DEVICES"] = selected_gpu

def safe_run(cmd):
    # Run the command
    try:
        subprocess.run(cmd, env=my_env, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")
        
# CUDA_VISIBLE_DEVICES=6 python render_dynamic.py -s /data/loyot/datasets/split-cookie --model_path output/hypernerf/split-cookie_fftpoly@20240130-144623/ --configs arguments/hypernerf/default.py --eval --skip_train --skip_test --ckpt_name chkpnt10000.pth
        
def run_excu(name, path, config, ckpt_name, save_npz=False):
    # first frame
    command = [
        'python', 'render_dynamic.py',
        '-s', f'{path}',
        '--model_path', f'output/{name}/',
        # '--iterations', '30000',
        '--configs', f'{config}',
        '--eval',
        '--skip_train',
        '--skip_test',
        '--ckpt_name', f'{ckpt_name}',
    ] + (
        ['--save_npz'] if save_npz else []
    )
    safe_run(command)


task_list = [
    # {
    #     'name': 'hypernerf/vrig_chicken_fftpoly@20240130-133120',
    #     'path': '/data/loyot/datasets/vrig-chicken',
    #     'config': "arguments/hypernerf/default.py",
    #     'ckpt_name': "chkpnt10000.pth",
    # },
    # {
    #     'name': 'hypernerf/interp_cut-lemon_fftpoly@20231212-024358',
    #     'path': '/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/interp_cut-lemon/cut-lemon1',
    #     'config': "arguments/hypernerf/default.py",
    #     'ckpt_name': "chkpnt10000.pth",
    # },
    # {
    #     'name': 'hypernerf/misc_americano_fftpoly@20231212-041251',
    #     'path': '/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/misc_americano/americano',
    #     'config': "arguments/hypernerf/default.py",
    #     'ckpt_name': "chkpnt10000.pth",
    # },
    # {
    #     'name': 'hypernerf/misc_espresso_fftpoly@20231212-034927',
    #     'path': '/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/misc_espresso/espresso',
    #     'config': "arguments/hypernerf/default.py",
    #     'ckpt_name': "chkpnt10000.pth",
    # },
    {
        'name': 'hypernerf/split-cookie_fftpoly@20240130-144623',
        'path': '/data/loyot/datasets/split-cookie',
        'config': "arguments/hypernerf/default.py",
        'ckpt_name': "chkpnt10000.pth",
        "save_npz": True,
    },
    # {
    #     'name': 'dnerf/jumpingjacks_fftpoly@20231222-233154',
    #     'path': '/home/loyot/workspace/SSD_1T/Datasets/NeRF/dynamic_data/jumpingjacks',
    #     'config': "arguments/dnerf/jumpingjacks.py",
    #     'ckpt_name': "chkpnt10000.pth",
    #     'save_npz': True,
    # },
    # {
    #     'name': 'dnerf/hook_fftpoly@20231227-183328',
    #     'path': '/home/loyot/workspace/SSD_1T/Datasets/NeRF/dynamic_data/hook',
    #     'config': "arguments/dnerf/hook.py",
    #     'ckpt_name': "chkpnt20000.pth",
    #     'save_npz': True,
    # },
]

for task in task_list:
    name = task['name']
    print(colored("Running: ", 'light_cyan'), f'{name}')
    run_excu(**task)
    sleep(5)
