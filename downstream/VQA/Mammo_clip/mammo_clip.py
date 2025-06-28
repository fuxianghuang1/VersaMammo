from torch import nn
import torch
from .modules import load_image_encoder
class Mammo_clip(nn.Module):
    def __init__(self, ckpt):
        super(Mammo_clip, self).__init__()
        self.config = ckpt["config"]["model"]["image_encoder"]
        self.image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"])
        image_encoder_weights = {}
        for k in ckpt["model"].keys():
            if k.startswith("image_encoder."):
                image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
        self.image_encoder.load_state_dict(image_encoder_weights, strict=True)
    def forward(self,x):
        image_features = self.image_encoder(x)
        # print(image_features.shape)
        # get [CLS] token for global representation (only for vision transformer)
        # global_features = image_features[:, 0]
        return image_features

# ckpt = torch.load('/home/jiayi/Baseline/Sotas/mammo-clip/b5-model-best-epoch-7.tar', map_location="cpu")
# # if ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
# #     args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
# # elif ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
# #     args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]
# model=Mammo_clip(ckpt)
# x=torch.rand(1,3,64,64)
# output=model(x)
# print(output.shape)