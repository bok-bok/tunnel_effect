#!/bin/bash


ID_DATA="yousuf_imagenet100"
OOD_DATA="ninco"
OOD_DATA_2="places"

GPU=0
GPU2=1
LR=3e-5

MODEL="vit_tiny_patch8_imagenet100_64"



python finetune_linear_probe.py \
--model $MODEL \
--pretrained_data $ID_DATA \
--data $ID_DATA \
--batch_size 512 \
--gpu1 $GPU \
--gpu2 $GPU2 \
--lr $LR

