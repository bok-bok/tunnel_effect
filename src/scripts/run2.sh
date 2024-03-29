#!/bin/bash


ID_DATA="yousuf_imagenet100"
OOD_DATA="ninco"
OOD_DATA_2="places"

GPU=0
GPU2=1

MODELS=("tvit_tiny_patch8_imagenet100_64" "tvit_tiny_patch8_imagenet100_224")

for MODEL in "${MODELS[@]}"
do
    for i in {1..3}
    do
        python main.py \
        --model $MODEL \
        --pretrained_data $ID_DATA \
        --data $ID_DATA \
        --batch_size 96 \
        --gpu1 $GPU \
        --gpu2 $GPU2
    done

    for i in {1..3}
    do
        python main.py \
        --model $MODEL \
        --pretrained_data $ID_DATA \
        --data $OOD_DATA \
        --batch_size 96 \
        --gpu1 $GPU \
        --gpu2 $GPU2
    done
done



done

# for i in {1..3}
# do
#     python main.py \
#     --model $MODEL \
#     --pretrained_data $ID_DATA \
#     --data $OOD_DATA_2 \
#     --batch_size 512 \
#     --gpu1 $GPU \
#     --gpu2 $GPU2
# done
