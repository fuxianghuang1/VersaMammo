'''
Dataset for training
Written by Whalechen
'''
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random 
from torchvision.transforms.functional import normalize
import math
import cv2
from time import time

class myRandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image_name, images, masks =  sample['imidx'], sample['image_name'], sample['images'],sample['masks']
        
        # random horizontal flip
        if random.random() >= self.prob:
            images = torch.flip(images,dims=[2])
            masks = torch.flip(masks,dims=[2])

        return {'imidx':imidx,'image_name':image_name, 'images':images, 'masks':masks}
class myRandomVFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image_name, images, masks =  sample['imidx'], sample['image_name'], sample['images'],sample['masks']
        
        # random horizontal flip
        if random.random() >= self.prob:
            images = torch.flip(images,dims=[1])
            masks = torch.flip(masks,dims=[1])

        return {'imidx':imidx,'image_name':image_name, 'images':images, 'masks':masks}

class myResize(object):
    def __init__(self,size=[224,224]):
        self.size = size
    def __call__(self,sample):
        imidx, image_name, images, masks =  sample['imidx'], sample['image_name'], sample['images'],sample['masks']

        # import time
        # start = time.time()

        images = torch.squeeze(F.interpolate(torch.unsqueeze(images,0),self.size,mode='bilinear'),dim=0)
        masks = torch.squeeze(F.interpolate(torch.unsqueeze(masks,0),self.size,mode='bilinear'),dim=0)

        return {'imidx':imidx,'image_name':image_name, 'images':images, 'masks':masks}
class myRandomCrop(object):
    def __init__(self,size=[320,320],prob=0.5):
        self.size = size
        self.prob=prob
    def __call__(self, sample):
        imidx, image_name, images, masks =  sample['imidx'], sample['image_name'], sample['images'],sample['masks']
        # if random.random() >= self.prob:
        resize = [[self.size[0],self.size[1]],[self.size[0]+32,self.size[1]+32], [self.size[0]+64,self.size[1]+64], [self.size[0]+96,self.size[1]+96], [self.size[0]+128,self.size[1]+128]][np.random.randint(0, 5)]
        images = torch.squeeze(F.interpolate(torch.unsqueeze(images,0),resize,mode='bilinear'),dim=0)
        masks = torch.squeeze(F.interpolate(torch.unsqueeze(masks,0),resize,mode='bilinear'),dim=0)
        H,W  = images.shape[1:]
        if H!=self.size[0] and W!=self.size[1]:
            offseth = np.random.randint(H-self.size[0])
            offsetw = np.random.randint(W-self.size[1])
        else:
            offseth=0
            offsetw=0
        images=images[:,offseth:offseth+self.size[0],offsetw:offsetw+self.size[1]]
        masks=masks[:,offseth:offseth+self.size[0],offsetw:offsetw+self.size[1]]
        return {'imidx':imidx,'image_name':image_name, 'images':images, 'masks':masks}

class myTranslate(object):
    def __init__(self , prob=0.5):
        self.prob=prob
    def __call__(self, sample):
        imidx, image_name, images, masks =  sample['imidx'], sample['image_name'], sample['images'],sample['masks']
        if random.random() >= self.prob:
            Ci, H, W = images.shape
            Cl, H, W = masks.shape
            tx = torch.FloatTensor(1).uniform_(0, W//2)
            ty = torch.FloatTensor(1).uniform_(0, H//2)
            affine_matrix = torch.tensor([[1, 0, 2*tx / W], [0, 1, 2*ty / H]], dtype=torch.float) 
            affine_matrix = affine_matrix.unsqueeze(0) 

            image_grid = F.affine_grid(affine_matrix, [1, Ci, H, W], align_corners=False)
            images = F.grid_sample(images.unsqueeze(0), image_grid, align_corners=False).squeeze(0)

            label_grid = F.affine_grid(affine_matrix, [1, Cl, H, W], align_corners=False)
            masks = F.grid_sample(masks.unsqueeze(0), label_grid, align_corners=False).squeeze(0)

        return {'imidx':imidx,'image_name':image_name, 'images':images, 'masks':masks}
class myRotate(object):
    def __init__(self , prob=0.5):
        self.prob=prob
    def __call__(self, sample):
        imidx, image_name, images, masks =  sample['imidx'], sample['image_name'], sample['images'],sample['masks']
        if random.random() >= self.prob:
            angle=torch.FloatTensor(1).uniform_(0, 360)
            angle = math.radians(angle) 
            cos = math.cos(angle)
            sin = math.sin(angle)
            affine_matrix = torch.tensor([[cos, -sin, 0], [sin, cos, 0]], dtype=torch.float)
            affine_matrix = affine_matrix.unsqueeze(0)

            image_grid = F.affine_grid(affine_matrix, images.unsqueeze(0).size(), align_corners=False)
            images = F.grid_sample(images.unsqueeze(0), image_grid, align_corners=False).squeeze(0)

            label_grid = F.affine_grid(affine_matrix, masks.unsqueeze(0).size(), align_corners=False)
            masks = F.grid_sample(masks.unsqueeze(0), label_grid, align_corners=False).squeeze(0)

        return {'imidx':imidx,'image_name':image_name, 'images':images, 'masks':masks}
class myNoise(object):
    def __init__(self , prob=0.5):
        self.prob=prob
    def __call__(self, sample):
        imidx, image_name, images, masks =  sample['imidx'], sample['image_name'], sample['images'],sample['masks']
        if random.random() >= self.prob:
            mean = torch.mean(images)
            std = torch.std(images)
            noise = torch.randn_like(images) * std + mean
            # Add noise to the image
            noisy_image = images + noise
            # Clip the pixel values to [0, 1]
            images = torch.clamp(noisy_image, 0, 1)

        return {'imidx':imidx,'image_name':image_name, 'images':images, 'masks':masks}

class myNormalize(object):
    def __init__(self, mean=[0.5], std=[1.0]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):

        imidx, image_name, images, masks =  sample['imidx'], sample['image_name'], sample['images'],sample['masks']

        images = normalize(images,self.mean,self.std)
    
        return {'imidx':imidx,'image_name':image_name, 'images':images, 'masks':masks}
def calculate_histogram(image):

    image = image[image >= 5]

    histogram, _ = np.histogram(image, bins=np.arange(257))
    return histogram

def calculate_ww_wc(histogram, low=0.3, up=0.7, medium=0.6):

    cdf = np.cumsum(histogram)
    cdf_normalized = cdf / cdf[-1]  
    

    low_bound = np.searchsorted(cdf_normalized, low)
    high_bound = np.searchsorted(cdf_normalized, up)
    

    wc = np.searchsorted(cdf_normalized, medium)
    
    ww = high_bound - low_bound
    
    return ww, wc

def _apply_windowing_torch(arr, window_width, window_center, voi_func='LINEAR', y_min=0, y_max=255):
    assert window_width > 0
    y_range = y_max - y_min
    arr = arr.float()

    if voi_func == 'LINEAR' or voi_func == 'LINEAR_EXACT':
        if voi_func == 'LINEAR':
            if window_width < 1:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than or "
                    "equal to 1 for a 'LINEAR' windowing operation")
            window_center -= 0.5
            window_width -= 1

        s = y_range / window_width
        b = (-window_center / window_width + 0.5) * y_range + y_min
        arr = arr * s + b
        arr = torch.clamp(arr, y_min, y_max)

    elif voi_func == 'SIGMOID':
        s = -4 / window_width
        arr = y_range / (1 + torch.exp((arr - window_center) * s)) + y_min
    else:
        raise ValueError(
            f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")
    return arr

def apply_windowing_torch(arr, window_width, window_center, voi_func='LINEAR', y_min=0, y_max=255):
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)

    arr = _apply_windowing_torch(arr,
                                 window_width=window_width,
                                 window_center=window_center,
                                 voi_func=voi_func,
                                 y_min=y_min,
                                 y_max=y_max)
    return arr

def im_preprocess(im,size):

    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    ##
    # im_lab = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)
    ##
    im_tensor = torch.tensor(im.copy(), dtype=torch.float32)
    im_tensor = torch.transpose(torch.transpose(im_tensor,1,2),0,1)
    if(len(size)<2):
        return im_tensor, im.shape[0:2]
    else:
        im_tensor = torch.unsqueeze(im_tensor,0)
        im_tensor = F.upsample(im_tensor, size, mode="bilinear")
        im_tensor = torch.squeeze(im_tensor,0)

    return im_tensor.type(torch.uint8)
def gt_preprocess(gt,size):

    if len(gt.shape) > 2:
        gt = gt[:, :, 0]
    gt_tensor = torch.unsqueeze(torch.tensor(gt, dtype=torch.uint8),0)

    if(len(size)<2):
        return gt_tensor.type(torch.uint8), gt.shape[0:2]
    else:
        gt_tensor = torch.unsqueeze(torch.tensor(gt_tensor, dtype=torch.float32),0)
        gt_tensor = F.upsample(gt_tensor, size, mode="bilinear")
        gt_tensor = torch.squeeze(gt_tensor,0)

    return gt_tensor.type(torch.uint8)
class myDataset(Dataset):
    def __init__(self, data_path,transform=None):
        # self.mode=mode
        # self.hypar=hypar
        self.data_path=data_path
        self.name_list=os.listdir(data_path)
        if transform!=None:
            self.transform=transforms.Compose(transform)
        else:
            self.transform=transform
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        
        images_path=os.path.join(self.data_path,self.name_list[idx],'img.pt')
        masks_path=os.path.join(self.data_path, self.name_list[idx],'mask.pt')
        # start=time()
        images=torch.load(images_path)
        masks=torch.load(masks_path)
      
        if images.shape[0]==1:
            images=images.repeat(3,1,1)
        
        images = torch.divide(images,255.0)

        masks = torch.divide(masks,255.0)

        sample={
            "imidx":idx,
            "image_name":self.name_list[idx],
            "images":images,
            "masks":masks
        }
        if self.transform!=None:
            sample = self.transform(sample)
        
        # sample["ori_images"]=images
        return sample
