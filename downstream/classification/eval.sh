#!/bin/bash

PYTHON_FILE="/classification/eval.py"

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CBIS-DDSM-breast'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Composition'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CBIS-DDSM-breast'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Bi-Rads'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CBIS-DDSM-finding'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Finding'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CBIS-DDSM-finding'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Pathology'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CBIS-DDSM-finding'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Subtlety'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CMMD'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Pathology'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CMMD'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Subtype'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='INbreast'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Composition'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='INbreast'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Bi-Rads'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='KAU-BCMD'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Bi-Rads'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[2]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='DMID-breast'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Composition'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='DMID-finding'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Pathology'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='MIAS-breast'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Composition'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='MIAS-finding'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Pathology'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='MIAS-finding'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Finding'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CSAW-M'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Masking'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='BMCD'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Composition'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='BMCD'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Bi-Rads'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CDD-CESM'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Composition'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CDD-CESM'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Bi-Rads'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CDD-CESM'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Pathology'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='DBT'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Pathology'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[1]/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='LAMIS'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Composition'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[1]/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='LAMIS'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Bi-Rads'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[1]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='LAMIS'/" $PYTHON_FILE
sed -i "s/hypar\['task'\]='.*'/hypar['task']='Pathology'/" $PYTHON_FILE
sed -i "s/hypar\['finetune'\]='.*'/hypar['finetune']='lp'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[1]/" $PYTHON_FILE
python $PYTHON_FILE &
sleep 30

