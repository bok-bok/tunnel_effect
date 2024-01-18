#!/bin/bash

GPU=0
BATCH_SIZE=128
EPOCHS=100
LRS=("1e-2" "1e-3" "1e-4" "1e-5")
WDS=("0" "1e-2" "1e-3" "1e-4")




SAMPLE_PER_CLASS=500

for LR in ${LRS[@]}
do
    for WD in ${WDS[@]}
    do
        python vgg11_100_samples_trainer.py \
            --sample_per_class $SAMPLE_PER_CLASS  \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --wd $WD \
            --epochs $EPOCHS \
            --gpu $GPU 
    done
done





