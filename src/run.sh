#!/bin/bash

MODEL="swin"
ID_DATA="imagenet"
OOD_DATA="places"

for i in {1..3}
do
    python main.py \
    --model $MODEL \
    --data $OOD_DATA \
    --batch_size 512 \
    --gpu1 1 \
    --gpu2 3  
done

for i in {1..3}
do
    python main.py \
    --model $MODEL \
    --data $ID_DATA \
    --batch_size 512 \
    --gpu1 1 \
    --gpu2 3  
done