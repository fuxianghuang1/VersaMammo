
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from efficientnet_custom import EfficientNet
from functools import partial
from torch import nn
from Mammo_clip.mammo_clip import Mammo_clip
import collections
from torchvision.ops.feature_pyramid_network import (
    LastLevelMaxPool,
    FeaturePyramidNetwork
)

class ViTBackboneWithFPN(nn.Module):
    def __init__(self, backbone, in_channels_list, out_channels, extra_blocks=None, norm_layer=nn.BatchNorm2d,input_size=512):
        super().__init__()
        self.input_size=input_size
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.backbone = backbone
        self.adapt=nn.Conv2d(self.backbone.out_channels,256,1)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x):
        if self.input_size==518:
            x=torch.nn.functional.interpolate(x,(518,518),mode='bilinear')
        if x.shape[2]==224:
            scales=[56,28,14,7]
        else:
            scales=[128,64,32,16]
        x=self.backbone(x)
        x=self.adapt(x)
        out = collections.OrderedDict()
        for i in range(4):  
            out[str(i)] = torch.nn.functional.interpolate(
                x,
                size=scales[i],
                mode="bilinear"
            )

        x = self.fpn(out)
        return x
class CNNBackboneWithFPN(nn.Module):
    def __init__(self, backbone, in_channels_list, out_channels, extra_blocks=None, norm_layer=nn.BatchNorm2d):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.backbone = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x):
        xs=self.backbone(x)
        out = collections.OrderedDict()
        out['0']=xs[0]
        out['1']=xs[1]
        out['2']=xs[2]
        out['3']=xs[3]
        x = self.fpn(out)
        return x
class ViTBackbone(torch.nn.Module):
    def __init__(self, checkpoint_path=None,ours=None,pretrained=True):
        super(ViTBackbone, self).__init__()
        self.ours=ours
        self.checkpoint_path=checkpoint_path
        if checkpoint_path!=None:
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

                self.model =ImageEncoderViT(
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
                self.model.load_state_dict(check_point,strict=False)
                print('LVM-Med vit-b loaded')  
                
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
                self.model=ImageEncoderViT(
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
                self.model.load_state_dict(new_state_dict, strict=False)  
                print('MedSAM vit-b loaded')
            elif 'mama' in checkpoint_path:
                from MaMA.load_weight import load_model
                self.model=load_model(checkpoint_path)
                print('mama vit-b loaded')
        self.group_norm=nn.GroupNorm(1,768)
        
    def forward(self, x):
        features = self.model(x)
        if self.checkpoint_path!=None:
            if 'lvmmed' in self.checkpoint_path or 'medsam' in self.checkpoint_path:
                features = self.group_norm(features)

        return features

def get_model(backbone_name="resnet50",pretrained=True,checkpoint_path=None,ours=None,finetune='lp',input_size=512):
    if backbone_name == "resnet50":
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
                
                return features 
        backbone = ResNet50Features(pretrained=True,checkpoint_path=checkpoint_path)
        backbone.out_channels = [256,512,1024,2048]
        in_channels_list=backbone.out_channels
        backbone=CNNBackboneWithFPN(backbone,in_channels_list,256)

    elif backbone_name == "efficientnet-b2":
        if pretrained==True:
            backbone = EfficientNet.from_pretrained('efficientnet-b2', num_classes=1)
        else:
            backbone = EfficientNet.from_name('efficientnet-b2')
        class EfficientNetBackbone(nn.Module):
            def __init__(self, backbone,checkpoint_path):
                super(EfficientNetBackbone, self).__init__()
                self.backbone = backbone
                self.encoder0 = nn.Sequential(self.backbone._conv_stem, self.backbone._bn0, self.backbone._swish)
                self.encoder1 = nn.Sequential(*self.backbone._blocks[:3])
                self.encoder2 = nn.Sequential(*self.backbone._blocks[3:6])
                self.encoder3 = nn.Sequential(*self.backbone._blocks[6:9])
                self.encoder4 = nn.Sequential(*self.backbone._blocks[9:13])
                self.encoder5 = nn.Sequential(*self.backbone._blocks[13:17])
                self.encoder6 = nn.Sequential(*self.backbone._blocks[17:22])
                self.encoder7 = nn.Sequential(*self.backbone._blocks[22:])
                if checkpoint_path:
                    ckpt = torch.load(checkpoint_path, map_location="cpu")
                    self.backbone=Mammo_clip(ckpt)
                    self.encoder0 = nn.Sequential(self.backbone.image_encoder._conv_stem, self.backbone.image_encoder._bn0, self.backbone.image_encoder._swish)
                    self.encoder1 = nn.Sequential(*self.backbone.image_encoder._blocks[:3])
                    self.encoder2 = nn.Sequential(*self.backbone.image_encoder._blocks[3:6])
                    self.encoder3 = nn.Sequential(*self.backbone.image_encoder._blocks[6:9])
                    self.encoder4 = nn.Sequential(*self.backbone.image_encoder._blocks[9:13])
                    self.encoder5 = nn.Sequential(*self.backbone.image_encoder._blocks[13:17])
                    self.encoder6 = nn.Sequential(*self.backbone.image_encoder._blocks[17:22])
                    self.encoder7 = nn.Sequential(*self.backbone.image_encoder._blocks[22:])

            def forward(self, x):
                enc0 = self.encoder0(x)
                enc1 = self.encoder1(enc0)
                enc2 = self.encoder2(enc1)
                enc3 = self.encoder3(enc2)
                enc4 = self.encoder4(enc3)
                enc5 = self.encoder5(enc4)
                enc6 = self.encoder6(enc5)
                enc7 = self.encoder7(enc6)
                return [enc3,enc4,enc6,enc7]
        backbone = EfficientNetBackbone(backbone,checkpoint_path)
        
        backbone.out_channels = [88,120,352,352]
        in_channels_list=backbone.out_channels
        backbone=CNNBackboneWithFPN(backbone,in_channels_list,256)
    elif backbone_name == "efficientnet-b5":
        if pretrained==True:
            backbone = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1)
        else:
            backbone = EfficientNet.from_name('efficientnet-b5')
        class EfficientNetBackbone(nn.Module):
            def __init__(self, backbone,checkpoint_path):
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
                if checkpoint_path:
                    ckpt = torch.load(checkpoint_path, map_location="cpu")
                    self.backbone=Mammo_clip(ckpt)
                    self.encoder0 = nn.Sequential(self.backbone.image_encoder._conv_stem, self.backbone.image_encoder._bn0, self.backbone.image_encoder._swish)
                    self.encoder1 = nn.Sequential(*self.backbone.image_encoder._blocks[:3])
                    self.encoder2 = nn.Sequential(*self.backbone.image_encoder._blocks[3:8])
                    self.encoder3 = nn.Sequential(*self.backbone.image_encoder._blocks[8:13])
                    self.encoder4 = nn.Sequential(*self.backbone.image_encoder._blocks[13:20])
                    self.encoder5 = nn.Sequential(*self.backbone.image_encoder._blocks[20:27])
                    self.encoder6 = nn.Sequential(*self.backbone.image_encoder._blocks[27:36])
                    self.encoder7 = nn.Sequential(*self.backbone.image_encoder._blocks[36:])

            def forward(self, x):
                enc0 = self.encoder0(x)
                enc1 = self.encoder1(enc0)
                enc2 = self.encoder2(enc1)
                enc3 = self.encoder3(enc2)
                enc4 = self.encoder4(enc3)
                enc5 = self.encoder5(enc4)
                enc6 = self.encoder6(enc5)
                enc7 = self.encoder7(enc6)
                return [enc3,enc4,enc6,enc7]
        backbone = EfficientNetBackbone(backbone,checkpoint_path)
        
        backbone.out_channels = [64,128,304,512]
        in_channels_list=backbone.out_channels
        backbone=CNNBackboneWithFPN(backbone,in_channels_list,256)
    elif backbone_name == "VersaMammo":

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

            def forward(self, x):
                enc0 = self.encoder0(x)
                enc1 = self.encoder1(enc0)
                enc2 = self.encoder2(enc1)
                enc3 = self.encoder3(enc2)
                enc4 = self.encoder4(enc3)
                enc5 = self.encoder5(enc4)
                enc6 = self.encoder6(enc5)
                enc7 = self.encoder7(enc6)
                return [enc3,enc4,enc6,enc7]
        backbone = EfficientNetBackbone(backbone)
        
        backbone.out_channels = [64,128,304,512]
        in_channels_list=backbone.out_channels
        backbone=CNNBackboneWithFPN(backbone,in_channels_list,256)
    elif backbone_name == "vit-b":
        backbone = ViTBackbone(checkpoint_path=checkpoint_path,ours=ours,pretrained=pretrained)
        backbone.out_channels = 768
        in_channels_list=[256]*4
        backbone=ViTBackboneWithFPN(backbone,in_channels_list,256,input_size=input_size)

    else:
        raise ValueError("Unsupported backbone")
    if finetune=='lp':
        for param in backbone.parameters():
            param.requires_grad = False
    anchor_generator = AnchorGenerator(
        sizes=((32,64,128),(32,64,128),(32,64,128),(32,64,128),(32,64,128)),
        aspect_ratios=((0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0))
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0','1','2','3'], output_size=7, sampling_ratio=2
    )
    
    if input_size==224:
        model = FasterRCNN(
            backbone,
            min_size=224,
            max_size=224,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
    else:
        model = FasterRCNN(
            backbone,
            min_size=512,
            max_size=512,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
    
    return model