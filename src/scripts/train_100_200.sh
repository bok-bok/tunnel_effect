#!/bin/bash

GPU=0
BATCH_SIZE=128
EPOCHS=30
LR=0.001
WD=0.0001


# SAMPLE_PER_CLASS=500

# python vgg11_100_samples_trainer.py \
#     --sample_per_class $SAMPLE_PER_CLASS  \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --wd $WD \
#     --epochs $EPOCHS \
#     --gpu $GPU 

SAMPLE_PER_CLASS=1000

python vgg11_100_samples_trainer.py \
    --sample_per_class $SAMPLE_PER_CLASS  \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --wd $WD \
    --epochs $EPOCHS \
    --gpu $GPU 
