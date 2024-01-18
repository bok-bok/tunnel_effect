#!/bin/bash


ID_DATA="imagenet"
OOD_DATA="ninco"
OOD_DATA_2="places"

GPU=3

MODEL="swin"
MODEL2="mae"

for i in {1..3}
do
    python main.py \
    --model $MODEL \
    --pretrained_data $ID_DATA \
    --data $OOD_DATA \
    --batch_size 512 \
    --gpu1 $GPU \
    --gpu2 $GPU
done




for i in {1..3}
do
    python main.py \
    --model $MODEL \
    --pretrained_data $ID_DATA \
    --data $OOD_DATA_2 \
    --batch_size 512 \
    --gpu1 $GPU \
    --gpu2 $GPU
done

for i in {1..3}
do
    python main.py \
    --model $MODEL \
    --pretrained_data $ID_DATA \
    --data $ID_DATA \
    --batch_size 512 \
    --gpu1 $GPU \
    --gpu2 $GPU
done


for i in {1..3}
do
    python main.py \
    --model $MODEL2 \
    --pretrained_data $ID_DATA \
    --data $OOD_DATA \
    --batch_size 512 \
    --gpu1 $GPU \
    --gpu2 $GPU
done




for i in {1..3}
do
    python main.py \
    --model $MODEL2 \
    --pretrained_data $ID_DATA \
    --data $OOD_DATA_2 \
    --batch_size 512 \
    --gpu1 $GPU \
    --gpu2 $GPU
done

for i in {1..3}
do
    python main.py \
    --model $MODEL2 \
    --pretrained_data $ID_DATA \
    --data $ID_DATA \
    --batch_size 512 \
    --gpu1 $GPU \
    --gpu2 $GPU
done
