#!/bin/bash

IMG_SIZE=224
PATCH_SIZE=8
BATCH_SIZE=128

LR=0.1
WEIGHT_DECAY=1e-3
EPOCHS=100

GPU=3

python VITTrainer.py \
    --img_size $IMG_SIZE \
    --patch_size $PATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --wd $WEIGHT_DECAY \
    --epochs $EPOCHS \
    --gpu $GPU \
    --num_labels 100