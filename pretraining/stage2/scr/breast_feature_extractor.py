import torchvision
from torch import nn

def load_image_encoder(model_name):  
    if model_name == 'enb5_rand':
        image_encoder = torchvision.models.efficientnet_b5(pretrained=False)
    elif model_name == 'enb5_in':
        image_encoder = torchvision.models.efficientnet_b5(pretrained=True)
    else:  
        raise ValueError(f"Unsupported model: {model_name}")
    
    image_encoder.classifier = nn.Identity()
    image_encoder.out_dim = 2048
    
    return image_encoder

class BreastFeatureExtract(nn.Module):
    def __init__(self, args):
        super(BreastFeatureExtract, self).__init__()     
        self.image_encoder_name = args.image_encoder_name
        self.image_encoder = load_image_encoder(self.image_encoder_name)
        # 添加 out_dim 属性
        self.out_dim = self.image_encoder.out_dim

    def forward(self, images):       
        image_feature = self.image_encoder(images)
        return image_feature

class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.classification_head = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.classification_head(x)       

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

