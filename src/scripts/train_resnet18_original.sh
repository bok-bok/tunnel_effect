#!/bin/bash

GPU=3
SIZE=32

BATCH_SIZE=128




python resnet18_resolution_original_trainer.py \
--size $SIZE \
--batch_size $BATCH_SIZE \
--gpu $GPU





