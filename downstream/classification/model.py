
import torch
import torch.nn as nn
from functools import partial
from Mammo_clip.mammo_clip import Mammo_clip
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
                    self.backbone=EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
                    image_encoder_weights = {}
                    for k in ckpt.keys():
                        if k.startswith("module.image_encoder."):
                            image_encoder_weights[".".join(k.split(".")[2:])] = ckpt[k]
                    self.backbone.load_state_dict(image_encoder_weights, strict=True)
                    print(checkpoint_path+' are loaded.')
                    self.backbone_features=2048
                elif 'b2' in checkpoint_path:
                    ckpt = torch.load(checkpoint_path, map_location="cpu")
                    self.backbone=Mammo_clip(ckpt)
                    print(checkpoint_path+' are loaded.')
                    self.backbone_features=1408
                elif 'b5' in checkpoint_path:
                    ckpt = torch.load(checkpoint_path, map_location="cpu")
                    self.backbone=Mammo_clip(ckpt)
                    print(checkpoint_path+' are loaded.')
                    self.backbone_features=2048
            elif backbone_name=='resnet50':
                if 'lvmmed' in checkpoint_path:
                    self.backbone = torchvision.models.resnet50(pretrained=True)
                    self.backbone.fc = nn.Identity()
                    pretrained_weight = torch.load(checkpoint_path)
                    self.backbone.load_state_dict(pretrained_weight, strict=False)
                    self.backbone_features=2048
            
            elif backbone_name=='MAMA':
                from MaMA.load_weight import load_model
                self.backbone=load_model(pretrained_model_path=checkpoint_path)
                self.backbone_features=768
        
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