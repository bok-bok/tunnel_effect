#!/bin/bash

SIZE=32
BATCH_SIZE=256

LR=0.1
WEIGHT_DECAY=0
EPOCHS=100


GPU=0

python trainer.py \
    --size $SIZE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --wd $WEIGHT_DECAY \
    --epochs $EPOCHS \
    --gpu $GPU \
