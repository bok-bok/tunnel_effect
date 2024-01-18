#!/bin/bash

TYPE=NC_1


# models=(vgg13_imagenet100_32 vgg13_imagenet100_64 vgg13_imagenet100_128 vgg13_imagenet100_224)

# for model in ${models[@]}; do
#     python NC_main.py \
#     --model ${model} \
#     --pretrained_data imagenet100 \
#     --input_size 5000 \
#     --type $TYPE
# done

# models=(resnet34 mlp)

# for model in ${models[@]}; do
#     python NC_main.py \
#     --model ${model} \
#     --pretrained_data cifar10 \
#     --input_size 10000 \
#     --type $TYPE
# done






models=(resnet50 resnet50_swav convnext vit mae mugs dino)


for model in ${models[@]}; do
    python NC_main.py \
    --model ${model} \
    --pretrained_data imagenet \
    --input_size 15000 \
    --type $TYPE
done

python NC_main.py \
    --model swin \
    --pretrained_data imagenet \
    --input_size 9000 \
    --type $TYPE


