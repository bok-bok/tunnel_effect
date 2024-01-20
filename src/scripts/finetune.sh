#!/bin/bash


ID_DATA="yousuf_imagenet100"
OOD_DATA="ninco"
OOD_DATA_2="places"

GPU=0
GPU2=1

MODEL="resnet18_imagenet100_32"

LR=1e-3





python finetune_linear_probe.py \
--model $MODEL \
--pretrained_data $ID_DATA \
--data $OOD_DATA \
--batch_size 96 \
--gpu1 $GPU \
--gpu2 $GPU2 \
--lr $LR \
--method "AVG" \
--patch_size 3 \
--normalization "True"


