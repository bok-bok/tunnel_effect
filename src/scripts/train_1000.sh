#!/bin/bash

CLASS_NUM=1000
BATCH_SIZE=128
 
LR=0.00001
WEIGHT_DECAY=0
EPOCHS=100


GPU=3

python vgg13_classes_trainer.py \
    --class_num $CLASS_NUM  \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --wd $WEIGHT_DECAY \
    --epochs $EPOCHS \
    --gpu $GPU \
