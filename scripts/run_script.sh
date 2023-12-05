# train the first frame of the scene
python train.py -s /home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/coffee_martini/images_split/0 --model_path output/coffee_martini/0

# train the rest of the scene
python train.py -s /home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/coffee_martini/images_split/0 --images /home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/coffee_martini/images_split/1/input --densify_until_iter 0 --train_rest_frame --model_path output/coffee_martini/1 --start_checkpoint /home/loyot/workspace/code/gaussian-splatting/output/coffee_martini/0/chkpnt30000.pth --iterations 2000

for i in {2..299}
do
    echo "Iteration $i"
    python train.py -s /home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/coffee_martini/images_split/0 --images /home/loyot/workspace/SSD_1T/Datasets/NeRF/3d_vedio_datasets/coffee_martini/images_split/1/input --densify_until_iter 0 --train_rest_frame --model_path output/coffee_martini/1 --start_checkpoint /home/loyot/workspace/code/gaussian-splatting/output/coffee_martini/0/chkpnt30000.pth --iterations 2000
done


# visluize the results
