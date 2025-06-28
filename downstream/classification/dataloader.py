'''
Dataset for training
Written by Whalechen
'''
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
import random 
from torchvision import transforms
from torchvision.transforms.functional import normalize
import math
from time import time

class myRandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image_name, images, labels =  sample['imidx'], sample['image_name'], sample['images'],sample['labels']
        
        # random horizontal flip
        if random.random() >= self.prob:
            images = torch.flip(images,dims=[2])

        return {'imidx':imidx,'image_name':image_name, 'images':images, 'labels':labels}
class myRandomVFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image_name, images, labels =  sample['imidx'], sample['image_name'], sample['images'],sample['labels']
        
        # random horizontal flip
        if random.random() >= self.prob:
            images = torch.flip(images,dims=[1])
        return {'imidx':imidx,'image_name':image_name, 'images':images, 'labels':labels}

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

        imidx, image_name, images, labels =  sample['imidx'], sample['image_name'], sample['images'],sample['labels']

        images = normalize(images,self.mean,self.std)
    
        return {'imidx':imidx,'image_name':image_name, 'images':images, 'labels':labels}
    
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
    def __init__(self, data_path, transform=None, label_mappings=None):
        self.label_mappings = label_mappings
        self.data_path = data_path
        self.name_list = os.listdir(data_path)
        self.valid_indices = []  

        for idx in range(len(self.name_list)):
            info_dict_path = os.path.join(self.data_path, self.name_list[idx], 'info_dict.pt')
            info_dict = torch.load(info_dict_path)

            valid_sample = False
            for task in self.label_mappings.keys():
                if task in info_dict:
                    valid_sample = True
                    break

            if valid_sample:
                self.valid_indices.append(idx)

        if transform is not None:
            self.transform = transforms.Compose(transform)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        images_path = os.path.join(self.data_path, self.name_list[real_idx], 'img.pt')
        info_dict_path = os.path.join(self.data_path, self.name_list[real_idx], 'info_dict.pt')

        images = torch.load(images_path)
        info_dict = torch.load(info_dict_path)

        if images.shape[0] == 1:
            images = images.repeat(3, 1, 1)
        images = torch.divide(images, 255.0)
        labels = {task: torch.tensor([self.label_mappings[task][info_dict[task][0] if isinstance(info_dict[task],list) else info_dict[task]]], dtype=torch.long) for task in self.label_mappings.keys()}
   
        sample = {
            "imidx": real_idx,
            "image_name": self.name_list[real_idx],
            "images": images,
            "labels": labels
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
