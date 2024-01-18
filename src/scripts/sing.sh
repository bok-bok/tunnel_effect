#!/bin/bash
SAMPLE_SIZE=15000

python compute_rank_main.py \
--model vgg11_imagenet_samples_200 \
--data imagenet \
--resolution 32 \
--num_classes 100 \
--input_size $SAMPLE_SIZE

python compute_rank_main.py \
--model vgg11_imagenet_samples_500 \
--data imagenet \
--resolution 32 \
--num_classes 100 \
--input_size $SAMPLE_SIZE

python compute_rank_main.py \
--model vgg11_imagenet_samples_1000 \
--data imagenet \
--resolution 32 \
--num_classes 100 \
--input_size $SAMPLE_SIZE
