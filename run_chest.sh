# Full: 89.08 
# Random Sample:89.79
# Remove CD: 84.57
# Remove flip: 81.02
# Original resize: 80.04
CUDA_VISIBLE_DEVICES=1 python3 train_network_dbt.py --dataset_path /data/usr/hd108/chest_xray_dataset --category chest --patch_size 128 --batch_size 300
