import subprocess
from termcolor import colored
import os


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


name = 'coffee_martini_poly_base_v18-15'
dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/coffee_martini/images_split/0'
config = "arguments/dynerf/default.py"

# name = 'flame_steak_poly_base_v18-10'
# dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/flame_steak/images_split/0'
# config = "arguments/dynerf/default.py"

# name = 'lego_poly_base_v18-11'
# dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/dynamic_data/lego'
# config = "arguments/dnerf/lego.py"

# name = 'vrig_chicken_poly_base_v18-14'
# dataset_path = '/home/loyot/workspace/SSD_1T/Datasets/NeRF/HyberNeRF/vrig_chicken/chicken'
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


# first frame
print(colored("Running: ", 'light_cyan'), f'frame: 0:')
command = [
    'python', 'train.py',
    '-s', f'{dataset_path}',
    '--model_path', f'output/{name}/',
    # '--iterations', '30000',
    '--config', f'{config}',
    '--test_iterations', '29999',
    '--eval'
]
safe_run(command)

    