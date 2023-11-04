import subprocess
from termcolor import colored
import os


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

name = 'espresso_poly_base_v17'
dataset_path = '/home/test/workspace/loyot/datasets/hyper_nerf/espresso'
config = "arguments/hypernerf/default.py"

# name = 'americano_poly_base_v17'
# dataset_path = '/home/test/workspace/loyot/datasets/hyper_nerf/americano'
# config = "arguments/hypernerf/default.py"

# name = 'chickchicken_poly_base_v17'
# dataset_path = '/home/test/workspace/loyot/datasets/hyper_nerf/chickchicken'
# config = "arguments/hypernerf/chickchicken.py"

# first frame
command = [
    'python', 'render_dynamic.py',
    '-s', f'{dataset_path}',
    '--model_path', f'output/{name}/',
    # '--iterations', '30000',
    '--configs', f'{config}',
    '--eval',
    '--skip_train',
    '--skip_test',
]
safe_run(command)
