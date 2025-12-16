import torch
import torchvision
from torch import nn
from models.efficientnet_custom import EfficientNet
from models import dinov2_vit
from models.dino_head import DINOHead
from functools import partial
from open_clip import create_model_from_pretrained, create_model_and_transforms, get_tokenizer
from transformers import AutoModel
from transformers import CLIPModel  

def dinov2_ckpt_wrapper(ckpt):
    # the dinov2 framework saved ckpt, e.g., teacher_checkpint.pth. 
    new_dict = {}
    for k, v in ckpt.items():
        k = '.'.join(k.split('.')[1:])
        new_dict[k] = v
    return new_dict

def build_vit_model(arch, pretrained=True, pretrained_path=' '):
    from models import dinov2_vit
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
    model = dinov2_vit.__dict__[arch](**vit_kwargs)
    if pretrained:        
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'best_birads_model' in pretrained_path:
            # model_state_dict = model.state_dict()
            new_state_dict = {}
            for key in checkpoint.keys():
                new_key = key.replace('module.feature_extractor.image_encoder.', '')
                new_state_dict[new_key] = checkpoint[key]
            msg = model.load_state_dict(new_state_dict, strict=False)

        elif 'teacher' in checkpoint:
        # if pretrained_path != '../pretrained_models/formated_vit-b_wo_reg.pth':
            # if you load the dinov2 automatically saved teacher_checkpoint.pth ckpt, call dinov2_ckpt_wrapper
            checkpoint = dinov2_ckpt_wrapper(checkpoint['teacher']) 
            msg = model.load_state_dict(checkpoint, strict=False)  
        
        else:
            msg = model.load_state_dict(checkpoint, strict=False)

        # msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    num_features = model.embed_dim
    return model, num_features

class MammoViTModel(nn.Module):
    def __init__(self, arch, pretrained=True, pretrained_path='../pretrained_models/formated_vit-b_wo_reg.pth'):
        super(MammoViTModel, self).__init__()

        self.vit_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'init_values': 1e-05,
            'ffn_layer': 'mlp',
            'block_chunks': 4,
            'qkv_bias': True,
            'proj_bias': True,
            'ffn_bias': True,
            'num_register_tokens': 0,
            'interpolate_offset': 0.1,
            'interpolate_antialias': False,
        }

        self.backbone = dinov2_vit.__dict__[arch](**self.vit_kwargs)
        self.dino_head = DINOHead(
            in_dim=768,
            hidden_dim=2048,
            bottleneck_dim=384,
            nlayers=3,
        )
        self.ibot_head = DINOHead(
            in_dim=768,
            hidden_dim=2048,
            bottleneck_dim=256,
            nlayers=3,
        )
        self.num_features = 768+384+256

        if pretrained:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        checkpoint = torch.load(pretrained_path)
        if pretrained_path != '../pretrained_models/formated_vit-b_wo_reg.pth':
            checkpoint = checkpoint['teacher']

        msg = self.load_state_dict(checkpoint, strict=False)
        print(msg)

    def forward(self, x):
        features = self.backbone(x)
        dino_output = self.dino_head(features)
        ibot_output = self.ibot_head(features)
        output = torch.cat((features, dino_output, ibot_output), dim=1)
        return output


def load_image_encoder(model_name):  
    if 'medsam' in model_name.lower():
        if model_name == 'medsam_vitb':
            from models.medsam_vit import ImageEncoderViT
            encoder_embed_dim=768
            encoder_depth=12
            encoder_num_heads=12
            encoder_global_attn_indexes=[2, 5, 8, 11]
            prompt_embed_dim = 256
            image_size = 1024
            vit_patch_size = 16           
            _image_encoder=ImageEncoderViT(
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
            check_point=torch.load('../pretrained_models/medsam_vit_b.pth',  map_location='cpu')
            new_state_dict = {}
            for key, value in check_point.items():
                new_key = key.replace('image_encoder.', '')
                new_state_dict[new_key] = value
            msg = _image_encoder.load_state_dict(new_state_dict, strict=False) 
            print(msg) 
            print('MedSAM vit-b loaded')
            _image_encoder.out_dim = 768
            print(_image_encoder)
 
    elif 'lvmmed' in model_name.lower():
        if model_name == 'lvmmed_vitb':
            from models.lvmmed_vit import ImageEncoderViT
            prompt_embed_dim = 256
            image_size = 1024
            vit_patch_size = 16
            encoder_embed_dim=768
            encoder_depth=12
            encoder_num_heads=12
            encoder_global_attn_indexes=[2, 5, 8, 11]

            _image_encoder =ImageEncoderViT(
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
            check_point = torch.load('../pretrained_models/lvmmed_vit.pth',  map_location='cpu')
            msg =_image_encoder.load_state_dict(check_point,strict=False)
            print(msg)
            print('LVM-Med vit-b loaded') 
            
            _image_encoder.out_dim = 768
            # print(_image_encoder)
        elif model_name == 'lvmmed_r50':
            _image_encoder = torchvision.models.resnet50(pretrained=True)
            pretrained_weight = torch.load('../pretrained_models/lvmmed_resnet.torch',  map_location='cpu')
            msg = _image_encoder.load_state_dict(pretrained_weight, strict=False)
            _image_encoder.fc = nn.Identity() #忽略FC
            print(msg)
            print('LVM-Med R50 loaded') 
            _image_encoder.out_dim = 2048
    elif 'mama' in model_name.lower():
        from MaMA.load_weight import load_model
        _image_encoder =load_model(pretrained_model_path='../pretrained_models/mama_embed_pretrained_40k_steps_last.ckpt')
        _image_encoder.out_dim = 768

        if model_name == 'vitb14rand':
            _image_encoder,_image_encoder.out_dim = build_vit_model(arch='vit_base', pretrained=False)

        elif model_name == 'vitb14dinov2':
            _image_encoder,_image_encoder.out_dim = build_vit_model(arch='vit_base', pretrained=True, pretrained_path='../pretrained_models/formated_vit-b_wo_reg.pth')

        elif model_name == 'vitb14dinov2mammo':
            _image_encoder,_image_encoder.out_dim = build_vit_model(arch='vit_base', pretrained=True, pretrained_path='../pretrained_models/teacher_checkpoint_474999.pth')   

        elif model_name == 'vitb14dinov2mammo1':
            _image_encoder,_image_encoder.out_dim = build_vit_model(arch='vit_base', pretrained=True, pretrained_path='../pretrained_models/teacher_checkpoint_324999.pth')
        
        elif model_name == 'vitb14birads':
            _image_encoder,_image_encoder.out_dim = build_vit_model(arch='vit_base', pretrained=True, pretrained_path='/mnt/data/hfx/code/downstream_tasks/model_vitb14_density_cl/model_epoch_4.pth')

        elif model_name == 'vitb14newmammo':
            _image_encoder,_image_encoder.out_dim = build_vit_model(arch='vit_base', pretrained=True, pretrained_path='../pretrained_models/new_checkpoint.pth')

        elif model_name == 'vitb14new_74999':
            _image_encoder,_image_encoder.out_dim = build_vit_model(arch='vit_base', pretrained=True, pretrained_path='../pretrained_models/new74999_checkpoint.pth')

        elif model_name == 'vitb14new_137499':
            _image_encoder,_image_encoder.out_dim = build_vit_model(arch='vit_base', pretrained=True, pretrained_path='../pretrained_models/new137499_checkpoint.pth')
     
        elif model_name == 'vitb14new_bd':
            _image_encoder,_image_encoder.out_dim = build_vit_model(arch='vit_base', pretrained=True, pretrained_path='/mnt/data/hfx/models/model_vitb14_bd_cl/best_birads_model.pth')
       
        elif model_name == 'vitb14new699999':
            _image_encoder,_image_encoder.out_dim = build_vit_model(arch='vit_base', pretrained=True, pretrained_path='../pretrained_models/699999_checkpoint.pth')

        elif model_name == 'vitb14dinov2mammocon':
            _image_encoder= MammoViTModel(arch='vit_base', pretrained=True, pretrained_path='../pretrained_models/teacher_checkpoint_474999.pth')   
            _image_encoder.out_dim = _image_encoder.num_features

        elif model_name == 'vitb14dinov2mammo1con':
            _image_encoder = MammoViTModel(arch='vit_base', pretrained=True, pretrained_path='../pretrained_models/teacher_checkpoint_324999.pth')
            _image_encoder.out_dim = _image_encoder.num_features

    elif model_name == 'EN-b2':
        _image_encoder = EfficientNet.from_pretrained('efficientnet-b2', num_classes=1)
        _image_encoder.out_dim = 1408

    elif model_name == 'mammoclip_b2':
        _image_encoder = EfficientNet.from_pretrained('efficientnet-b2', num_classes=1)
        _image_encoder.out_dim = 1408
        ckpt = torch.load('../pretrained_models/b2-model-best-epoch-10.tar', map_location="cpu")
        image_encoder_weights = {}
        for k in ckpt["model"].keys():
            if k.startswith("image_encoder."):
                image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
        _image_encoder.load_state_dict(image_encoder_weights, strict=True)   

    elif model_name == 'EN_b5':
        _image_encoder = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1)
        _image_encoder.out_dim = 2048

    elif model_name == 'mammoclip_b5':
        _image_encoder = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1)
        _image_encoder.out_dim = 2048
        ckpt = torch.load('../pretrained_models/b5-model-best-epoch-7.tar', map_location="cpu")
        image_encoder_weights = {}
        for k in ckpt["model"].keys():
            if k.startswith("image_encoder."):
                image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
        _image_encoder.load_state_dict(image_encoder_weights, strict=True)
    
    elif model_name == 'versamammo':
        _image_encoder = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1)
        _image_encoder.out_dim = 2048
        ckpt = torch.load('../pretrained_models/ENB5.pth', map_location="cpu")
        image_encoder_weights = {}
        for k in ckpt.keys():
            if k.startswith("module.image_encoder."):
                image_encoder_weights[".".join(k.split(".")[2:])] = ckpt[k]
        msg = _image_encoder.load_state_dict(image_encoder_weights, strict=False)
        print('our local model', msg)

    else:  
        raise ValueError(f"Unsupported model: {model_name}") 
    return _image_encoder 

class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.classification_head = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.classification_head(x) 
       
class BreastFeatureExtract(nn.Module):
    def __init__(self, args):
        super(BreastFeatureExtract, self).__init__()     
        self.image_encoder_name = args.image_encoder_name
        self.image_encoder = load_image_encoder(self.image_encoder_name)
    def forward(self, images):       
        image_feature = self.image_encoder(images)
        return image_feature

class BreastClassifier(nn.Module):
    def __init__(self, args):
        super(BreastClassifier, self).__init__()     
        self.image_encoder_name = args.image_encoder_name
        self.image_encoder = load_image_encoder(self.image_encoder_name)
        self.birads_classifier = LinearClassifier(input_size=self.image_encoder.out_dim, num_classes=args.birads_n_class)
        self.density_classifier = LinearClassifier(input_size=self.image_encoder.out_dim, num_classes=args.density_n_class)

    def forward(self, images):       
        image_feature = self.image_encoder(images)
        birads_out = self.birads_classifier(image_feature)
        density_out = self.density_classifier(image_feature)
        return birads_out, density_out, image_feature

