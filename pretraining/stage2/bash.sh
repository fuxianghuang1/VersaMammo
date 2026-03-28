#!/bin/bash

# Set visible GPU devices
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Launch training in background
nohup python main.py \
    --data_dir /mnt/data/hfx \
    --output_dir /mnt/data/hfx/versamammo_stage2 \ 
    > versamammo_stage2.log 2>&1 &
