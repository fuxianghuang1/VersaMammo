
import torch
import torch.nn as nn
from functools import partial
import torchvision
from efficientnet_custom import EfficientNet

class MultiTaskModel(nn.Module):
    """['resnet50', 'efficientnet_b2', 'vit_base_patch16_224']"""
    """['vitb14imagenet','vitb14dinov2mammo','vitb14dinov2mammo1']"""
    def __init__(self, backbone_name, data_info, checkpoint_path=None,ours=None,finetune='lp'):
        super(MultiTaskModel, self).__init__()
        for task, classes in data_info.items():
            self.class_num={task:len(classes)}
        if checkpoint_path:
            
            if backbone_name=='efficientnet':
                if 'VersaMammo' in checkpoint_path:
                    ckpt = torch.load(checkpoint_path, map_location="cpu")
                    self.backbone=EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
                    image_encoder_weights = {}
                    for k in ckpt.keys():
                        if k.startswith("module.image_encoder."):
                            image_encoder_weights[".".join(k.split(".")[2:])] = ckpt[k]
                    self.backbone.load_state_dict(image_encoder_weights, strict=True)
                    print(checkpoint_path+' are loaded.')
                    self.backbone_features=2048
                
        
        if finetune=='lp':
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(self.backbone_features, len(classes)) 
            for task, classes in data_info.items()
        })
    
    def forward(self, x):
        features = self.backbone(x)
        outputs = {task: classifier(features) for task, classifier in self.classifiers.items()}
        return outputs