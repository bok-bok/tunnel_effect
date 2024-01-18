#!/bin/bash

GPU=1
BATCH_SIZE=128
EPOCHS=300
LRS=("1e-2" "1e-3" "1e-4" "1e-5")
WDS=("0" "1e-2" "1e-3" "1e-4")



CLASS_NUM=50
for LR in ${LRS[@]}
do
    for WD in ${WDS[@]}
    do
        python vgg13_classes_trainer.py \
            --class_num $CLASS_NUM  \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --wd $WD \
            --epochs $EPOCHS \
            --gpu $GPU 
    done
done


