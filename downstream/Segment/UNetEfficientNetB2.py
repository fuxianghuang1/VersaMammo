import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet
from Mammo_clip.mammo_clip import Mammo_clip

class UNetEfficientNetB2(nn.Module):
    def __init__(self, num_classes=1, pretrained=True,checkpoint_path=None):
        super(UNetEfficientNetB2, self).__init__()
        if pretrained==True:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b2')
        elif pretrained==False:
            self.backbone = EfficientNet.from_name('efficientnet-b2')
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

        self.pool = nn.MaxPool2d(2, 2)
        self.decoder8 = self.decoder_block(352, 176, 352)

        self.decoder7 = self.decoder_block(704, 176, 352)

        self.decoder6 = self.decoder_block(704, 104, 208)

        self.decoder5 = self.decoder_block(416, 60, 120)

        self.decoder4 = self.decoder_block(240, 44, 88)
        
        self.decoder3 = self.decoder_block(176, 24, 48)

        self.decoder2 = self.decoder_block(96, 12, 24)
        
        self.decoder1 = self.decoder_block(48, 16, 32)
        
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
    def decoder_block(self, in_channels, mid_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        enc7 = self.encoder7(enc6)
        
        dec8 = self.pool(enc7)
        dec8 = self.decoder8(dec8)

        dec7 = F.interpolate(dec8,enc7.shape[2:],mode='bilinear')
        dec7 = torch.cat([dec7, enc7], dim=1)
        dec7 = self.decoder7(dec7)

        dec6 = F.interpolate(dec7,enc6.shape[2:],mode='bilinear')
        dec6 = torch.cat([dec6, enc6], dim=1)
        dec6 = self.decoder6(dec6)

        dec5 = F.interpolate(dec6,enc5.shape[2:],mode='bilinear')
        dec5 = torch.cat([dec5, enc5], dim=1)
        dec5 = self.decoder5(dec5)

        dec4 = F.interpolate(dec5,enc4.shape[2:],mode='bilinear')
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = F.interpolate(dec4,enc3.shape[2:],mode='bilinear')
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = F.interpolate(dec3,enc2.shape[2:],mode='bilinear')
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = F.interpolate(dec2,enc1.shape[2:],mode='bilinear')
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(F.interpolate(self.final_conv(dec1), x.shape[2:], mode='bilinear'))