#!/bin/bash

# Set visible GPU devices
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Add DINOv2 module to Python path
export PYTHONPATH=dinov2

# Launch distributed training in background
nohup torchrun --nproc_per_node=4 \
    dinov2/train/train.py \
    --config-file dinov2/configs/train/versamammo_vitb.yaml \
    --output-dir saved_dir \
    > dinov2/saved_dir.log 2>&1 &
