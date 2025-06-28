#!/bin/bash

PYTHON_FILE="/Detection/main.py"

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CBIS-DDSM'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ".*"/os.environ['CUDA_VISIBLE_DEVICES'] = '5'/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30
#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='INbreast'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ".*"/os.environ['CUDA_VISIBLE_DEVICES'] = '5'/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30
#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='VinDr-Mammo'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ".*"/os.environ['CUDA_VISIBLE_DEVICES'] = '5'/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30
