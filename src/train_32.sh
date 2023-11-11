#!/bin/bash

SIZE=32
BATCH_SIZE=128

LR=0.1
MOMENTUM=0.9
WEIGHT_DECAY=0.005
EPOCHS=164


GPU=0

python trainer.py \
    --size $SIZE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --wd $WEIGHT_DECAY \
    --momentum $MOMENTUM \
    --epochs $EPOCHS \
    --milestones 22 50 \
    --gpu $GPU \
