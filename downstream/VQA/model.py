
import torch
import torch.nn as nn
from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import ViltConfig
import os
from functools import partial
from Mammo_clip.mammo_clip import Mammo_clip
import torchvision
from Ours.models.image_encoder import load_image_encoder
from efficientnet_custom import EfficientNet

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
        if checkpoint_path:
            if backbone_name=='vit_base_patch16_224':
                if 'lvmmed' in checkpoint_path:
                    from lvmmed_vit import ImageEncoderViT
                    prompt_embed_dim = 256
                    image_size = 1024
                    vit_patch_size = 16
                    image_embedding_size = image_size // vit_patch_size
                    encoder_embed_dim=768
                    encoder_depth=12
                    encoder_num_heads=12
                    encoder_global_attn_indexes=[2, 5, 8, 11]

                    self.backbone =ImageEncoderViT(
                            depth=encoder_depth,
                            embed_dim=encoder_embed_dim,
                            img_size=image_size,
                            mlp_ratio=4,
                            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                            num_heads=encoder_num_heads,
                            patch_size=vit_patch_size,
                            qkv_bias=True,
                            use_rel_pos=True,
                            use_abs_pos = False,
                            global_attn_indexes=encoder_global_attn_indexes,
                            window_size=14,
                            out_chans=prompt_embed_dim,
                        )
                    check_point = torch.load(checkpoint_path)
                    self.backbone.load_state_dict(check_point,strict=False)
                    print('LVM-Med vit-b loaded')  
                    self.backbone_features=768
                elif 'medsam' in checkpoint_path:
                    from medsam_vit import ImageEncoderViT
                    encoder_embed_dim=768
                    encoder_depth=12
                    encoder_num_heads=12
                    encoder_global_attn_indexes=[2, 5, 8, 11]
                    prompt_embed_dim = 256
                    image_size = 1024
                    vit_patch_size = 16
                    image_embedding_size = image_size // vit_patch_size
                    self.backbone=ImageEncoderViT(
                        depth=encoder_depth,
                        embed_dim=encoder_embed_dim,
                        img_size=image_size,
                        mlp_ratio=4,
                        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                        num_heads=encoder_num_heads,
                        patch_size=vit_patch_size,
                        qkv_bias=True,
                        use_rel_pos=True,
                        global_attn_indexes=encoder_global_attn_indexes,
                        window_size=14,
                        out_chans=prompt_embed_dim,
                    )
                    check_point=torch.load(checkpoint_path)
                    new_state_dict = {}
                    for key, value in check_point.items():
                        new_key = key.replace('image_encoder.', '')
                        new_state_dict[new_key] = value
                    self.backbone.load_state_dict(new_state_dict, strict=False)  
                    
                    self.backbone_features=768
                    print('MedSAM vit-b loaded')
            elif backbone_name=='efficientnet':
                if 'VersaMammo' in checkpoint_path:
                    ckpt = torch.load(checkpoint_path, map_location="cpu")
                    backbone=EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
                    image_encoder_weights = {}
                    for k in ckpt.keys():
                        if k.startswith("module.image_encoder."):
                            image_encoder_weights[".".join(k.split(".")[2:])] = ckpt[k]
                    backbone.load_state_dict(image_encoder_weights, strict=True)
                    print(checkpoint_path+' are loaded.')
                    class EfficientNetBackbone(nn.Module):
                        def __init__(self, backbone):
                            super(EfficientNetBackbone, self).__init__()
                            self.backbone = backbone
                            self.encoder0 = nn.Sequential(self.backbone._conv_stem, self.backbone._bn0, self.backbone._swish)
                            self.encoder1 = nn.Sequential(*self.backbone._blocks[:3])
                            self.encoder2 = nn.Sequential(*self.backbone._blocks[3:8])
                            self.encoder3 = nn.Sequential(*self.backbone._blocks[8:13])
                            self.encoder4 = nn.Sequential(*self.backbone._blocks[13:20])
                            self.encoder5 = nn.Sequential(*self.backbone._blocks[20:27])
                            self.encoder6 = nn.Sequential(*self.backbone._blocks[27:36])
                            self.encoder7 = nn.Sequential(*self.backbone._blocks[36:])
                            self.fusion=Fusion(64+128+304+512)
                        def forward(self, x):
                            enc0 = self.encoder0(x)
                            enc1 = self.encoder1(enc0)
                            enc2 = self.encoder2(enc1)
                            enc3 = self.encoder3(enc2)
                            enc4 = self.encoder4(enc3)
                            enc5 = self.encoder5(enc4)
                            enc6 = self.encoder6(enc5)
                            enc7 = self.encoder7(enc6)
                            output=self.fusion([enc3,enc4,enc6,enc7])
                            return output
                    self.backbone = EfficientNetBackbone(backbone)
                    
                    self.backbone_features=256
                elif 'b2' in checkpoint_path:
                    class EfficientNetBackbone(nn.Module):
                        def __init__(self, checkpoint_path):
                            super(EfficientNetBackbone, self).__init__()
                            ckpt = torch.load(checkpoint_path, map_location="cpu")
                            self.backbone=Mammo_clip(ckpt)
                            print(checkpoint_path+' are loaded.')
                            self.encoder0 = nn.Sequential(self.backbone.image_encoder._conv_stem, self.backbone.image_encoder._bn0, self.backbone.image_encoder._swish)
                            self.encoder1 = nn.Sequential(*self.backbone.image_encoder._blocks[:3])
                            self.encoder2 = nn.Sequential(*self.backbone.image_encoder._blocks[3:6])
                            self.encoder3 = nn.Sequential(*self.backbone.image_encoder._blocks[6:9])
                            self.encoder4 = nn.Sequential(*self.backbone.image_encoder._blocks[9:13])
                            self.encoder5 = nn.Sequential(*self.backbone.image_encoder._blocks[13:17])
                            self.encoder6 = nn.Sequential(*self.backbone.image_encoder._blocks[17:22])
                            self.encoder7 = nn.Sequential(*self.backbone.image_encoder._blocks[22:])
                            self.fusion=Fusion(88+120+352+352)
                        def forward(self, x):
                            enc0 = self.encoder0(x)
                            enc1 = self.encoder1(enc0)
                            enc2 = self.encoder2(enc1)
                            enc3 = self.encoder3(enc2)
                            enc4 = self.encoder4(enc3)
                            enc5 = self.encoder5(enc4)
                            enc6 = self.encoder6(enc5)
                            enc7 = self.encoder7(enc6)
                            output=self.fusion([enc3,enc4,enc6,enc7])
                            return output
                    self.backbone = EfficientNetBackbone(checkpoint_path)
                    
                    self.backbone_features=256
                elif 'b5' in checkpoint_path:
                    class EfficientNetBackbone(nn.Module):
                        def __init__(self, checkpoint_path):
                            super(EfficientNetBackbone, self).__init__()
                            
                            ckpt = torch.load(checkpoint_path, map_location="cpu")
                            self.backbone=Mammo_clip(ckpt)
                            print(checkpoint_path+' are loaded.')
                            self.encoder0 = nn.Sequential(self.backbone.image_encoder._conv_stem, self.backbone.image_encoder._bn0, self.backbone.image_encoder._swish)
                            self.encoder1 = nn.Sequential(*self.backbone.image_encoder._blocks[:3])
                            self.encoder2 = nn.Sequential(*self.backbone.image_encoder._blocks[3:8])
                            self.encoder3 = nn.Sequential(*self.backbone.image_encoder._blocks[8:13])
                            self.encoder4 = nn.Sequential(*self.backbone.image_encoder._blocks[13:20])
                            self.encoder5 = nn.Sequential(*self.backbone.image_encoder._blocks[20:27])
                            self.encoder6 = nn.Sequential(*self.backbone.image_encoder._blocks[27:36])
                            self.encoder7 = nn.Sequential(*self.backbone.image_encoder._blocks[36:])
                            self.fusion=Fusion(64+128+304+512)
                        def forward(self, x):
                            enc0 = self.encoder0(x)
                            enc1 = self.encoder1(enc0)
                            enc2 = self.encoder2(enc1)
                            enc3 = self.encoder3(enc2)
                            enc4 = self.encoder4(enc3)
                            enc5 = self.encoder5(enc4)
                            enc6 = self.encoder6(enc5)
                            enc7 = self.encoder7(enc6)
                            output=self.fusion([enc3,enc4,enc6,enc7])
                            return output
                    self.backbone = EfficientNetBackbone(checkpoint_path)
                    self.backbone_features=256
            elif backbone_name=='resnet50':
                if 'lvmmed' in checkpoint_path:
                    class ResNet50Features(nn.Module):
                        def __init__(self, pretrained=True, checkpoint_path=None):
                            super(ResNet50Features, self).__init__()
                            
                            backbone = torchvision.models.resnet50(pretrained=pretrained)
                            backbone.fc = nn.Identity()

                            if checkpoint_path:
                                backbone.load_state_dict(torch.load(checkpoint_path), strict=False)
                                print(checkpoint_path + ' are loaded.')

                            self.stage1 = nn.Sequential(
                                backbone.conv1,
                                backbone.bn1,
                                backbone.relu,
                                backbone.maxpool
                            )
                            
                            self.stage2 = backbone.layer1
                            self.stage3 = backbone.layer2
                            self.stage4 = backbone.layer3
                            self.stage5 = backbone.layer4 
                            self.fusion=Fusion(256+512+1024+2048)
                        
                        def forward(self, x):
                            features = []
                            x = self.stage1(x)
                            
                            x = self.stage2(x)
                            features.append(x) 
                            
                            x = self.stage3(x)
                            features.append(x)  
                            
                            x = self.stage4(x)
                            features.append(x) 
                            
                            x = self.stage5(x)
                            features.append(x)  
                            output=self.fusion(features)
                            return output  

                    self.backbone = ResNet50Features(pretrained=True,checkpoint_path=checkpoint_path)
                    self.backbone_features = 256
            elif 'mama' in checkpoint_path:
                from MaMA.load_weight import load_model
                self.backbone=load_model(checkpoint_path)
                print('mama vit-b loaded')
                self.backbone_features = 768

        elif ours:
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


