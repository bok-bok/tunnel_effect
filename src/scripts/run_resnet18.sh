#!/bin/bash


ID_DATA="imagenet100"
OOD_DATA="ninco"
OOD_DATA_2="places"
# ID_DATA="cifar10"
# OOD_DATA="cifar100"

GPU=0
GPU2=0

# MODELS=("resnet18_imagenet100_no_residual_32" "resnet18_imagenet100_no_residual_64" "resnet18_imagenet100_no_residual_128" "resnet18_imagenet100_no_residual_224")
MODELS=("resnet18_imagenet100_aug_32" "resnet18_imagenet100_aug_64" "resnet18_imagenet100_aug_128" "resnet18_imagenet100_aug_224")
for MODEL in ${MODELS[@]}
    do
        for i in {1..3}
        do
            python main.py \
            --model $MODEL \
            --pretrained_data $ID_DATA \
            --data $OOD_DATA_2 \
            --batch_size 512 \
            --gpu1 $GPU \
            --gpu2 $GPU2
        done

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