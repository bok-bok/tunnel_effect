#!/bin/bash


TYPE=NC_2

models=(resnet34 mlp)

for model in ${models[@]}; do
    python NC_main.py \
    --model ${model} \
    --pretrained_data cifar10 \
    --input_size 1000 \
    --type $TYPE
done

models=(vgg13_imagenet100_32 vgg13_imagenet100_64 vgg13_imagenet100_128 vgg13_imagenet100_224)

for model in ${models[@]}; do
    python NC_main.py \
    --model ${model} \
    --pretrained_data imagenet100 \
    --input_size 1000 \
    --type $TYPE
done

