#!/bin/bash

PYTHON_FILE="/VQA/main.py"

#
sed -i "s/hypar\['question topic'\]='.*'/hypar['question topic']='View'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ".*"/os.environ['CUDA_VISIBLE_DEVICES'] = '6'/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30


sed -i "s/hypar\['question topic'\]='.*'/hypar['question topic']='Laterality'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ".*"/os.environ['CUDA_VISIBLE_DEVICES'] = '7'/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30


sed -i "s/hypar\['question topic'\]='.*'/hypar['question topic']='Bi-Rads'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ".*"/os.environ['CUDA_VISIBLE_DEVICES'] = '7'/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30


sed -i "s/hypar\['question topic'\]='.*'/hypar['question topic']='Pathology'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ".*"/os.environ['CUDA_VISIBLE_DEVICES'] = '7'/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

# #
sed -i "s/hypar\['question topic'\]='.*'/hypar['question topic']='ACR'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ".*"/os.environ['CUDA_VISIBLE_DEVICES'] = '7'/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30


sed -i "s/hypar\['question topic'\]='.*'/hypar['question topic']='Subtlety'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ".*"/os.environ['CUDA_VISIBLE_DEVICES'] = '7'/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30



sed -i "s/hypar\['question topic'\]='.*'/hypar['question topic']='Masking potential'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ".*"/os.environ['CUDA_VISIBLE_DEVICES'] = '7'/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

