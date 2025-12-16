import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, Compose, Resize, InterpolationMode, ToTensor
from collections import OrderedDict
from typing import Optional, Tuple
from PIL import Image

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        
        return x[0], x[1:].transpose(0, 1)

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def stem(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_cls, x_tokens = self.attnpool(x)

        visual_output = dict.fromkeys(["image_features", "mim_loss"], None)
        visual_output.update({
            'image_features': x_cls,
            'image_token_features': x_tokens
        })

        return visual_output

def image_transform(
        image_size: int,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        fill_color: int = 0,
):
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    mean = mean or (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
    std = std or (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
    normalize = Normalize(mean=mean, std=std)

    transforms = [
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
    ]
    transforms.extend([
        ToTensor(),
        normalize,
    ])
    return Compose(transforms)

class PMC_CLIP_ImageEncoder(nn.Module):
    """
    PMC-CLIP图像特征提取器
    基于ModifiedResNet架构，专门用于医学图像特征提取
    """
    
    def __init__(
            self,
            layers=[3, 4, 6, 3],
            output_dim=768,
            heads=32,
            image_size=224,
            width=64
    ):
        super().__init__()
        
        self.visual = ModifiedResNet(
            layers=layers,
            output_dim=output_dim,
            heads=heads,
            image_size=image_size,
            width=width
        )
        
        # 图像预处理
        self.preprocess = image_transform(image_size=image_size)
        
    def encode_image(self, image):
        """
        编码图像
        
        Args:
            image: 输入图像，可以是PIL图像或已经预处理过的tensor
            
        Returns:
            dict: 包含图像特征的字典
                - image_features: 全局图像特征 [batch_size, output_dim]
                - image_token_features: 图像token特征 [batch_size, num_tokens, output_dim]
        """
        return self.visual(image)
    
    def preprocess_image(self, image):
        """
        预处理图像
        
        Args:
            image: PIL图像或图像路径
            
        Returns:
            torch.Tensor: 预处理后的图像tensor
        """
        if isinstance(image, str):
            # 如果是文件路径，加载图像
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            # 已经是PIL图像
            pass
        else:
            raise ValueError("Input must be a PIL image or file path")
            
        return self.preprocess(image)
    
    def forward(self, images):
        """
        前向传播
        
        Args:
            images: 输入图像tensor [batch_size, 3, H, W]
            
        Returns:
            dict: 包含图像特征的字典
        """
        return self.encode_image(images)

# 使用示例
def create_pmc_clip_image_encoder(ckpt_path=None, device='cuda'):
    """
    创建PMC-CLIP图像编码器
    
    Args:
        ckpt_path: 预训练权重路径
        device: 设备
        
    Returns:
        PMC_CLIP_ImageEncoder: 图像编码器实例
    """
    model = PMC_CLIP_ImageEncoder()
    
    if ckpt_path:
        # 加载预训练权重
        ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
        
        # 移除'module.'前缀
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith('module.'):
                name = k[7:]
            elif k.startswith('visual.'):
                name = k
            else:
                continue
            new_ckpt[name] = v
            
        # 加载视觉部分权重
        visual_ckpt = {k.replace('visual.', ''): v for k, v in new_ckpt.items() if k.startswith('visual.')}
        model.visual.load_state_dict(visual_ckpt, strict=False)
    
    model.to(device)
    model.eval()
    return model

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = create_pmc_clip_image_encoder()
    
    # 示例使用
    from PIL import Image
    import torch
    
    # 加载和预处理图像
    image_path = "/home/fuxianghuang/code/downstream_tasks/enhanced_tsne_results/lvmmed_r50_enhanced_tsne.png"  # 替换为你的图像路径
    image = Image.open(image_path).convert('RGB')
    processed_image = model.preprocess_image(image)
    
    # 添加批次维度
    input_tensor = processed_image.unsqueeze(0).to('cuda')
    
    # 提取特征
    with torch.no_grad():
        features = model(input_tensor)
    
    print("图像特征形状:", features['image_features'].shape)  # [1, 768]
    print("图像token特征形状:", features['image_token_features'].shape)  # [1, 49, 768]