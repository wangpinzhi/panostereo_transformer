#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --batch_size 1\
                --checkpoint kitti_toy_eval\
                --num_workers 2\
                --eval\
                --dataset kitti_toy\
                --dataset_directory /home/data/wangpinzhi/Omni_Transformer/Deep360\
                --resume /home/data/wangpinzhi/Omni_Transformer/stereo_transformer/run/deep360/deep360_ft/experiment_10/model.pth.tar