#!/bin/bash

SIZE=224
BATCH_SIZE=512

LR=0.1
MOMENTUM=0.9
WEIGHT_DECAY=0.003
EPOCHS=45


GPU=3

python trainer.py \
    --size $SIZE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --wd $WEIGHT_DECAY \
    --momentum $MOMENTUM \
    --epochs $EPOCHS \
    --milestones 20 30 40 \
    --gpu $GPU \
