'''
Dataset for training
Written by Whalechen
'''
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io
import torch.nn.functional as F
from PIL import Image
# import clip_surgery
from tqdm import tqdm
import random 
from torchvision.transforms import functional as TF
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import math
import cv2
from time import time

class myRandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image_name, images, targets =  sample['imidx'], sample['image_name'], sample['images'],sample['targets']
        
        # random horizontal flip
        if random.random() >= self.prob:
            images = torch.flip(images,dims=[2])
            _, height, width = images.shape
            bbox = targets['boxes']
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            targets['boxes'] = bbox

        return {'imidx':imidx,'image_name':image_name, 'images':images, 'targets':targets}
class myRandomVFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image_name, images, targets =  sample['imidx'], sample['image_name'], sample['images'],sample['targets']
        
        # random horizontal flip
        if random.random() >= self.prob:
            images = torch.flip(images,dims=[1])
            _, height, width = images.shape
            bbox = targets['boxes']
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            targets['boxes'] = bbox

        return {'imidx':imidx,'image_name':image_name, 'images':images, 'targets':targets}

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
            affine_matrix = torch.tensor([[1, 0, 2*tx / W], [0, 1, 2*ty / H]], dtype=torch.float)  # 归一化平移量
            affine_matrix = affine_matrix.unsqueeze(0)  # 批处理维度

            # 创建仿射网格，并确保align_corners的设置与grid_sample中一致
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
            angle = math.radians(angle)  # 角度转弧度
            cos = math.cos(angle)
            sin = math.sin(angle)
            affine_matrix = torch.tensor([[cos, -sin, 0], [sin, cos, 0]], dtype=torch.float)
            affine_matrix = affine_matrix.unsqueeze(0)  # 批处理维度

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

        imidx, image_name, images, targets =  sample['imidx'], sample['image_name'], sample['images'],sample['targets']

        images = normalize(images,self.mean,self.std)
    
        return {'imidx':imidx,'image_name':image_name, 'images':images, 'targets':targets}
    
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
        bboxes_path=os.path.join(self.data_path, self.name_list[idx],'bboxes.pt')
        # start=time()
        images=torch.load(images_path)
        bboxes=torch.load(bboxes_path)
        if images.shape[0]>1:
            images = images.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
            # 使用 OpenCV 将彩色图像转换为灰度图像
            images = cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)
        else:
            images=images.squeeze(0).numpy()
        if images.dtype != np.uint8:
            images = cv2.normalize(images, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        images = clahe.apply(images)
        images = torch.from_numpy(images).float().unsqueeze(0).repeat(3,1,1)
        images = torch.divide(images,255.0)
        # bboxes = torch.divide(bboxes,255.0)
        # print(time()-start)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.tensor([1]*len(target["boxes"]), dtype=torch.int64)  # 假设只有一个类，标签为1
        sample={
            "imidx":idx,
            "image_name":self.name_list[idx],
            "images":images,
            "targets":target
        }
        if self.transform!=None:
            sample = self.transform(sample)
        # sample["ori_images"]=images
        return sample["imidx"],sample["image_name"],sample["images"],sample["targets"]
