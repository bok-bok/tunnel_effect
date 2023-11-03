#!/bin/bash

MODEL="dinov2"
ID_DATA="imagenet"
OOD_DATA="places"

for i in {1..2}
do
    python main.py \
    --model $MODEL \
    --data $OOD_DATA \
    --batch_size 512 \
    --gpu1 0 \
    --gpu2 0  
done

# for i in {1..3}
# do
#     python main.py \
#     --model $MODEL \
#     --data $ID_DATA \
#     --batch_size 64 \
#     --gpu1 0 \
#     --gpu2 0


# done