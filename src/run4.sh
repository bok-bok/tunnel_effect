#!/bin/bash

MODEL="convnextv2"
ID_DATA="imagenet"
OOD_DATA="places"


for i in {1..3}
do
    python main.py \
    --model $MODEL \
    --data $ID_DATA \
    --batch_size 512 \
    --gpu1 2 \
    --gpu2 2  
done