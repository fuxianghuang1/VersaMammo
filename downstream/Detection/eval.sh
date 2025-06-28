#!/bin/bash

PYTHON_FILE="/Detection/eval.py"

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CBIS-DDSM'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[3]/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30
#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='INbreast'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[3]/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30
#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='VinDr-Mammo'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[3]/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30
