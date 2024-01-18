#!/bin/bash


TYPE=NC_2
# models=(vit swin resnet50 convnext resnet50_swav mae mugs dino)
models=(convnext)

for model in ${models[@]}; do
    python NC_main.py \
    --model ${model} \
    --pretrained_data imagenet \
    --input_size 1000 \
    --type $TYPE
done

# python NC_main.py \
#     --model swin \
#     --pretrained_data imagenet \
#     --input_size 1000 \
#     --type $TYPE

# models=(resnet34 mlp)

# for model in ${models[@]}; do
#     python NC_main.py \
#     --model ${model} \
#     --pretrained_data cifar10 \
#     --input_size 1000 \
#     --type NC_4
# done

# models=(vgg13_imagenet100_32 vgg13_imagenet100_64 vgg13_imagenet100_128 vgg13_imagenet100_224)

# for model in ${models[@]}; do
#     python NC_main.py \
#     --model ${model} \
#     --pretrained_data imagenet100 \
#     --input_size 1000 \
#     --type $TYPE
# done

