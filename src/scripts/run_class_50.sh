#!/bin/bash


ID_DATA="imagenet"
OOD_DATA="ninco"
OOD_DATA_2="places"
BATCH_SIZE=256

MODEL="vgg13_imagenet_class_50"
GPU=0


for i in {1..3}
do
    python vgg13_classes_run.py \
    --model $MODEL \
    --pretrained_data $ID_DATA \
    --data $OOD_DATA_2 \
    --batch_size $BATCH_SIZE \
    --gpu1 $GPU \
    --gpu2 $GPU
done

for i in {1..3}
do
    python vgg13_classes_run.py \
    --model $MODEL \
    --pretrained_data $ID_DATA \
    --data $ID_DATA \
    --batch_size $BATCH_SIZE \
    --gpu1 $GPU \
    --gpu2 $GPU
done



for i in {1..3}
do
    python vgg13_classes_run.py \
    --model $MODEL \
    --pretrained_data $ID_DATA \
    --data $OOD_DATA \
    --batch_size $BATCH_SIZE \
    --gpu1 $GPU \
    --gpu2 $GPU
done
