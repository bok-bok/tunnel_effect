#!/bin/bash
GPU=1
BATCH_SIZE=128
EPOCHS=30

LR=0.001
WD=0.0001

CLASS_NUM=10
python vgg11_classes_trainer.py \
    --class_num $CLASS_NUM  \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --wd $WD \
    --epochs $EPOCHS \
    --gpu $GPU 

CLASS_NUM=50
python vgg11_classes_trainer.py \
    --class_num $CLASS_NUM  \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --wd $WD \
    --epochs $EPOCHS \
    --gpu $GPU 


