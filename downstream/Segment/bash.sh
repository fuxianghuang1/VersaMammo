#!/bin/bash

PYTHON_FILE="/Segment/main.py"

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CBIS-DDSM-split'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ".*"/os.environ['CUDA_VISIBLE_DEVICES'] = '7'/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30
#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='INbreast-split'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ".*"/os.environ['CUDA_VISIBLE_DEVICES'] = '7'/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30

