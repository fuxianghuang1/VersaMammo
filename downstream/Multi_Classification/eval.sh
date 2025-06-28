#!/bin/bash

PYTHON_FILE="/Multi_Classification/eval.py"

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='VinDr-Mammo-finding'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Finding'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[1]/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CMMD'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Finding'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[1]/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='INbreast'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Finding'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[1]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='DMID-finding'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Finding'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[1]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

