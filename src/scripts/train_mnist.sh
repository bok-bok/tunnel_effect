#!/bin/bash

GPU=0
SIZE=32

BATCH_SIZE=256
EPOCHS=20
LRS=("1e-2" "1e-3" "1e-4" "1e-5")
WDS=("0" "1e-2" "1e-3" "1e-4")




for LR in ${LRS[@]}
do
    for WD in ${WDS[@]}
    do
        python small_trainer.py \
        --batch_size $BATCH_SIZE \
        --dataset mnist \
        --model resnet34 \
        --lr $LR \
        --wd $WD \
        --epochs $EPOCHS \
        --gpu $GPU
    done
done





