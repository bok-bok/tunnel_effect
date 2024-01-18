#!/bin/bash

GPU=1
SIZE=32

BATCH_SIZE=256
EPOCHS=70

LR=1e-3
WD=1e-4

SIZES=(128 224)

for SIZE in ${SIZES[@]}
    do
        python resnet18_resolution_trainer.py \
        --size $SIZE \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --wd $WD \
        --epochs $EPOCHS \
        --gpu $GPU
    done





