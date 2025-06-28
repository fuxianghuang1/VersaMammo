import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class UNetResNet50(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, checkpoint_path=None):
        super(UNetResNet50, self).__init__()
        
        # Load ResNet50 backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Identity()
        if checkpoint_path:
            self.backbone.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.encoder0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
        self.encoder1 = self.backbone.layer1
        self.encoder2 = self.backbone.layer2
        self.encoder3 = self.backbone.layer3
        self.encoder4 = self.backbone.layer4

        self.pool = nn.MaxPool2d(2, 2)
        self.decoder5 = self.decoder_block(2048, 512, 2048)

        # Decoder layers
        self.decoder4 = self.decoder_block(4096, 256, 1024)
        
        self.decoder3 = self.decoder_block(2048, 128, 512)

        self.decoder2 = self.decoder_block(1024, 64, 256)
        
        self.decoder1 = self.decoder_block(512, 32, 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
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
        
        dec5 = self.pool(enc4)
        dec5 = self.decoder5(dec5)

        # Decoder with skip connections
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