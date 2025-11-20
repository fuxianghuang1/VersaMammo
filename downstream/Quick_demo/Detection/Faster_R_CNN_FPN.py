
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from efficientnet_custom import EfficientNet
from functools import partial
from torch import nn
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

def get_model(backbone_name="resnet50",pretrained=True,checkpoint_path=None,ours=None,finetune='lp',input_size=512):
    if backbone_name == "VersaMammo":

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