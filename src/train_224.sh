#!/bin/bash

SIZE=224
BATCH_SIZE=128

LR=0.1
WEIGHT_DECAY=1e-3
EPOCHS=100


GPU=3

python trainer.py \
    --size $SIZE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --wd $WEIGHT_DECAY \
    --epochs $EPOCHS \
    --gpu $GPU \
