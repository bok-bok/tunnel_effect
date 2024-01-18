#!/bin/bash

GPU=1
SIZE=224

BATCH_SIZE=256
EPOCHS=50
LRS=("1e-4")
WDS=("0" "1e-2" "1e-3" "1e-4")




for LR in ${LRS[@]}
do
    for WD in ${WDS[@]}
    do
        python resnet18_resolution_trainer.py \
        --size $SIZE \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --wd $WD \
        --epochs $EPOCHS \
        --gpu $GPU
    done
done





