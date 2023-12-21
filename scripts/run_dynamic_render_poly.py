import subprocess
from termcolor import colored
import os
from time import sleep

selected_gpu = '0'
my_env = os.environ.copy()
my_env["CUDA_VISIBLE_DEVICES"] = selected_gpu

def safe_run(cmd):
    # Run the command
    try:
        subprocess.run(cmd, env=my_env, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")
        
        
def run_excu(name, path, config, ckpt_name):
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
    ]
    safe_run(command)

# name = 'flame_steak_poly_base_v10'
# dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/flame_steak/images_split/0'

# name = 'lego_poly_base_v12'
# dataset_path = '/home/test/workspace/loyot/datasets/hyper_nerf/vrig-chicken'
# config = "arguments/dnerf/lego.py"

# name = 'vrig_chicken_poly_base_v17'
# dataset_path = '/home/test/workspace/loyot/datasets/hyper_nerf/vrig-chicken'
# config = "arguments/hypernerf/default.py"

# name = 'split-cookie_poly_base_v17'
# dataset_path = '/home/test/workspace/loyot/datasets/hyper_nerf/split-cookie'
# config = "arguments/hypernerf/default.py"

# name = 'espresso_poly_base_v17'
# dataset_path = '/home/test/workspace/loyot/datasets/hyper_nerf/espresso'
# config = "arguments/hypernerf/default.py"

# name = 'americano_poly_base_v17'
# dataset_path = '/home/test/workspace/loyot/datasets/hyper_nerf/americano'
# config = "arguments/hypernerf/default.py"

# name = 'chickchicken_poly_base_v17'
# dataset_path = '/home/test/workspace/loyot/datasets/hyper_nerf/chickchicken'
# config = "arguments/hypernerf/chickchicken.py"

# name = 'dnerf/jumpingjacks_fftpoly@20231208-040752'
# dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/dynamic_data/jumpingjacks'
# config = "arguments/dnerf/jumpingjacks.py"

# name = 'dynerf/flame_steak_fftpoly@20231210-154735'
# dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/flame_steak'
# config = "arguments/dynerf/default.py"
# ckpt_name = "chkpnt30000.pth"


task_list = [
    # {
    #     'name': 'hypernerf/interp_chickchicken_fftpoly@20231212-024358',
    #     'path': '/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/interp_chickchicken/chickchicken',
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
    {
        'name': 'hypernerf/misc_espresso_fftpoly@20231212-034927',
        'path': '/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/misc_espresso/espresso',
        'config': "arguments/hypernerf/default.py",
        'ckpt_name': "chkpnt10000.pth",
    },
    # {
    #     'name': 'hypernerf/misc_split-cookie_fftpoly@20231212-024358',
    #     'path': '/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/misc_split-cookie/split-cookie',
    #     'config': "arguments/hypernerf/default.py",
    #     'ckpt_name': "chkpnt10000.pth",
    # },
]

for task in task_list:
    name = task['name']
    print(colored("Running: ", 'light_cyan'), f'{name}')
    run_excu(task["name"], task["path"], task["config"], task["ckpt_name"])
    sleep(5)
