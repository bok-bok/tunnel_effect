#!/bin/bash

SIZE=128
BATCH_SIZE=256

LR=0.1
MOMENTUM=0.9
WEIGHT_DECAY=0.003
EPOCHS=40


GPU=2

python trainer.py \
    --size $SIZE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --wd $WEIGHT_DECAY \
    --momentum $MOMENTUM \
    --epochs $EPOCHS \
    --milestones 15 25 35 \
    --gpu $GPU \
