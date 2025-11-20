
import torch
import torch.nn as nn
from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import ViltConfig
import os
from functools import partial
import torchvision
from VersaMammo.models.image_encoder import load_image_encoder

class ViltWithBackbone(ViltForQuestionAnswering):
    def __init__(self, config, vision_backbone,class_num):
        super().__init__(config)
        self.backbone = vision_backbone  
        self.question_topic, self.classnum = list(class_num.items())[0]
        self.vqa_outputs = nn.Linear(config.hidden_size, self.classnum)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, pixel_values=None, labels=None):
        self.vilt.embeddings.patch_embeddings=self.backbone
        outputs = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
        )
        if self.question_topic=='Abnormality':
            logits = {self.question_topic:torch.sigmoid(self.vqa_outputs(outputs.last_hidden_state[:, 0, :]))}
        else:
            logits = {self.question_topic:self.vqa_outputs(outputs.last_hidden_state[:, 0, :])}
        
        loss = None
        if labels is not None:
            labels=torch.tensor(labels).to(input_ids.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits[self.question_topic].view(-1, self.classnum), labels.view(-1))
        return logits, loss
class Backbone(nn.Module):
    def __init__(self,backbone,hidden_dim):
        super().__init__()
        self.backbone=backbone
        self.connector = nn.Sequential(
            nn.Conv2d(hidden_dim, 256,1),
            nn.GELU(),
            nn.Conv2d(256, 768,1)
        )

    def forward(self, x):
        x=self.backbone(x)
        x=self.connector(x)
        return x
class Fusion(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.fusion = nn.Conv2d(hidden_dim, 256,1)
    def forward(self, x):
        target_size = x[-2].shape[2:]
        interpolated_features = []
        for feature in x:
            interpolated_feature = torch.nn.functional.interpolate(feature, size=target_size, mode='bilinear')
            interpolated_features.append(interpolated_feature)

        concatenated_features = torch.cat(interpolated_features, dim=1)
        return self.fusion(concatenated_features)
class MultiTaskModel(nn.Module):
    """['resnet50', 'efficientnet_b2', 'vit_base_patch16_224']"""
    """['vitb14imagenet','vitb14dinov2mammo','vitb14dinov2mammo1']"""
    def __init__(self, backbone_name, data_info, checkpoint_path=None,ours=None,finetune='lp'):
        super(MultiTaskModel, self).__init__()
        for task, classes in data_info.items():
            self.question_topic=task
            class_num={task:len(classes)}
        if ours:
            self.backbone=load_image_encoder(ours)
            print(ours+' are loaded.')
            self.backbone_features=768
       
        if finetune=='lp':
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        vilt_config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

        vision_backbone=Backbone(self.backbone,self.backbone_features)
        self.model = ViltWithBackbone(config=vilt_config, vision_backbone=vision_backbone,class_num=class_num)
    
    def forward(self, image,question,labels,size=(224,224)):
        encoding = self.processor(image, question, return_tensors="pt")
        pixel_values = torch.nn.functional.interpolate(encoding['pixel_values'],size,mode='bilinear').to(self.model.device)
        input_ids = encoding['input_ids'].to(self.model.device)
        attention_mask = encoding['attention_mask'].to(self.model.device)
        
        logits, loss = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,labels=labels)
        return logits, loss


