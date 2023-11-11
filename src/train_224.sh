#!/bin/bash

SIZE=224
BATCH_SIZE=512

LR=0.1
MOMENTUM=0.9
WEIGHT_DECAY=0.005
EPOCHS=128


GPU=3

python trainer.py \
    --size $SIZE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --wd $WEIGHT_DECAY \
    --momentum $MOMENTUM \
    --epochs $EPOCHS \
    --milestones 40 75 90 \
    --gpu $GPU \
