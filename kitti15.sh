#!/usr/bin/env bash
set -x
DATAPATH="/YOUR_DATA_PATH"
CUDA_VISIBLE_DEVICES=0 python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
    --epochs 800 --lrepochs "200,400,600:2" \
    --logdir /YOUR_LOG_DIR \
    --loadckpt /YOUR_PRETRAINED_MODEL_PATH \
    --batch_size 6 --test_batch_size 4 \
    --lr 0.0005 --c 4 --mag 3 --channels_3d 8 --growth_rate 4 1 --channel_CRP 16 \
    --ckpt_start_epoch 800 --summary_freq 9999999
