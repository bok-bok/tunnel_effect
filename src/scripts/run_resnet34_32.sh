#!/bin/bash


ID_DATA="cifar10"
OOD_DATA="cifar100"
# OOD_DATA_2="places"

GPU=0
GPU2=0

MODELS=("resnet34_original_aug_final" )
for MODEL in ${MODELS[@]}
    do
        for i in {1..3}
        do
            python main.py \
            --model $MODEL \
            --pretrained_data $ID_DATA \
            --data $OOD_DATA \
            --batch_size 512 \
            --gpu1 $GPU \
            --gpu2 $GPU2
        done




        for i in {1..3}
        do
            python main.py \
            --model $MODEL \
            --pretrained_data $ID_DATA \
            --data $ID_DATA \
            --batch_size 512 \
            --gpu1 $GPU \
            --gpu2 $GPU2
        done


    done