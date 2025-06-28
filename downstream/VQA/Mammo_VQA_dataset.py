import torch
from PIL import Image
import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
# base_dir='/home/jiayi/MammoVQA'
# sys.path.append(os.path.join(base_dir, 'Eval'))
class MammoVQA_image(torch.utils.data.Dataset):
    def __init__(self, loaded_data,label_mappings,type='single'):
        self.loaded_data = loaded_data
        # self.base_dir = base_dir
        # self.vis_processor = vis_processor
        self.label_mappings=label_mappings
        self.type=type

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        sample = self.loaded_data[idx]
        image_path = '/home/jiayi/MammoVQA'+sample['Path']
        image=Image.open(image_path).convert('RGB')
        question=sample['Question']
        label=sample['Answer']
        question_topic=sample['Question topic']
        if self.type=='single':
            label = self.label_mappings[question_topic][sample['Answer']]
        else:
            
            label_mapping = self.label_mappings[question_topic]
            label = np.zeros(len(label_mapping), dtype=np.float32)  # 初始化一个全零的 label 向量
            for finding in sample['Answer']:
                if finding in label_mapping:  # 如果答案在映射中，设置相应位置为 1.0
                    label[label_mapping[finding]] = 1.0

        ids=sample['ID']
        ## question-answering-score
        # qas_prompt= build_prompt(sample,score_type='question_answering_score')
        # cs_prompt = build_prompt(sample,score_type='certain_score')
        # image = Image.open(image_path).convert('RGB')
        # image = self.vis_processor(image)
        return image, question, label, ids
def custom_collate_fn(batch):
    images, questions, labels, ids = zip(*batch)
    return list(images), list(questions), list(labels), list(ids)