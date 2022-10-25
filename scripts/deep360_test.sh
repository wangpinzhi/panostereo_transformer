#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --batch_size 1\
                --checkpoint deep360_test\
                --num_workers 1\
                --eval\
                --use_shpereconv\
                --dataset deep360\
                --dataset_directory /home/data/wangpinzhi/Omni_Transformer/Deep360\
                --resume /home/data/wangpinzhi/Omni_Transformer/stereo_transformer/kitti_finetuned_model.pth.tar
 