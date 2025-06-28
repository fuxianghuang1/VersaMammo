#import timm
import torch
from torch import nn
# from models import ResNet
import vision_transformer as vits
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir=current_dir.split('FM_downstream')[0]+'FM_downstream'
def dinov2_ckpt_wrapper(ckpt):
    # the dinov2 framework saved ckpt, e.g., teacher_checkpint.pth. 
    new_dict = {}
    for k, v in ckpt.items():
        k = '.'.join(k.split('.')[1:])
        new_dict[k] = v
    return new_dict

def build_vit_model(arch, pretrained=True, pretrained_path='formated_vit-b_wo_reg.pth'):
    vit_kwargs = dict(
        img_size=224,
        patch_size=14,
        init_values=1e-05,
        ffn_layer='mlp',
        block_chunks=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        num_register_tokens=0,
        interpolate_offset=0.1,
        interpolate_antialias=False,
    )
    model = vits.__dict__[arch](**vit_kwargs)#only image encoder
    if pretrained:        
        checkpoint = torch.load(pretrained_path)
        if pretrained_path.split('/')[-1] != 'formated_vit-b_wo_reg.pth':
            # if you load the dinov2 automatically saved teacher_checkpoint.pth ckpt, call dinov2_ckpt_wrapper
            checkpoint = dinov2_ckpt_wrapper(checkpoint['teacher'])   
        
        msg = model.load_state_dict(checkpoint, strict=False)
        print('msg:',msg)
    num_features = model.embed_dim
    return model, num_features

def load_image_encoder(model_name):  
    if 'resnet' in model_name.lower():  
        _image_encoder = ResNet(model_name)  # resnet50  
 
    elif 'vitb14' in model_name.lower():

        if model_name == 'vitb14rand':
            _image_encoder,_image_encoder.out_dim = build_vit_model(arch='vit_base', pretrained=False)

        elif model_name == 'vitb14versamammo':
            _image_encoder,_image_encoder.out_dim = build_vit_model(arch='vit_base', pretrained=True, pretrained_path=f'{base_dir}/Sotas/VersaMammo/ViT/teacher_checkpoint_324999.pth')

    else:  
        raise ValueError(f"Unsupported model: {model_name}") 
    return _image_encoder 
