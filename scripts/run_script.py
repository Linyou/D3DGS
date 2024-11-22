import os
from termcolor import colored

if __name__ == "__main__":
    for i in range(2, 300):
        print(
            colored("Running: ", 'light_cyan'), 
            f'frame: {i}:'
        )
        first_frame = 0
        last_frame = i - 1
        current_frame = i
        cmd = f"python train.py -s /home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/coffee_martini/images_split/{first_frame} --images /home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/coffee_martini/images_split/{current_frame}/input --densify_until_iter 0 --train_rest_frame --model_path output/coffee_martini/{current_frame} --start_checkpoint /home/loyot/workspace/code/gaussian-splatting/output/coffee_martini/{last_frame}/chkpnt2000.pth --iterations 2000"
        os.system(cmd)