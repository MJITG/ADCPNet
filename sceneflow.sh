#!/usr/bin/env bash
set -x
DATAPATH="/YOUR_DATA_PATH"
CUDA_VISIBLE_DEVICES=1 python main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --epochs 40 --lrepochs "10,12,14,16:1" \
    --logdir /YOUR_LOG_DIR \
    --batch_size 6 --test_batch_size 4 \
    --summary_freq 9999999 \
    --lr 0.0005 --c 4 --mag 3 --channels_3d 8 --growth_rate 4 1 --channel_CRP 16 \
    --ckpt_start_epoch 30