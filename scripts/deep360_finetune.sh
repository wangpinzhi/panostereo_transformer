#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --epochs 400\
                --batch_size 1\
                --checkpoint deep360_ft\
                --num_workers 4\
                --dataset deep360\
                --dataset_directory /home/data/wangpinzhi/Omni_Transformer/Deep360\
                --ft\
                --use_shpereconv\
                --resume /home/data/wangpinzhi/Omni_Transformer/stereo_transformer/sceneflow_pretrained_model.pth.tar
