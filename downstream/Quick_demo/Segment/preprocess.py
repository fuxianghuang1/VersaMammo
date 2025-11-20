import os
import torch
import numpy as np
import cv2
import pydicom as pdcm
from tqdm import tqdm
import torchvision.transforms.functional as F
from skimage import io

def preprocess(input_path,output_path,output_size=[224,224]):
    for folder_name in tqdm(os.listdir(input_path)):
        folder_path = os.path.join(input_path, folder_name)
        target_path=os.path.join(output_path, folder_name)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        if os.path.isdir(folder_path):
            img_path = os.path.join(folder_path, 'img.jpg')
            mask_path = os.path.join(folder_path, 'mask.png')
            bboxes_path = os.path.join(folder_path, 'bboxes.npy')

            if os.path.exists(img_path) and os.path.exists(mask_path) and os.path.exists(bboxes_path):
                img=io.imread(img_path)

                mask = io.imread(mask_path)

                bboxes = np.load(bboxes_path, allow_pickle=True)

                original_size = img.shape
                img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)

                h_scale = output_size[0] / original_size[0]
                w_scale = output_size[1] / original_size[1]
                bboxes = [(x1 * w_scale, y1 * h_scale, x2 * w_scale, y2 * h_scale) for (x1, y1, x2, y2) in bboxes]

     
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
                mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) 
                bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)

       
                img_save_path = os.path.join(target_path, 'img.pt')
                mask_save_path = os.path.join(target_path, 'mask.pt')
                bboxes_save_path = os.path.join(target_path, 'bboxes.pt')
                if not os.path.exists(img_save_path):
                    torch.save(img_tensor, img_save_path)
                if not os.path.exists(mask_save_path):
                    torch.save(mask_tensor, mask_save_path)
                if not os.path.exists(bboxes_save_path):
                    torch.save(bboxes_tensor, bboxes_save_path)
                
            elif os.path.exists(img_path) and os.path.exists(bboxes_path):

                img=io.imread(img_path)

                bboxes = np.load(bboxes_path, allow_pickle=True)

                original_size = img.shape
                img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)

                h_scale = output_size[0] / original_size[0]
                w_scale = output_size[1] / original_size[1]
                bboxes = [(x1 * w_scale, y1 * h_scale, x2 * w_scale, y2 * h_scale) for (x1, y1, x2, y2) in bboxes]


                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) 
                bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)

                img_save_path = os.path.join(target_path, 'img.pt')
                bboxes_save_path = os.path.join(target_path, 'bboxes.pt')
                if not os.path.exists(img_save_path):
                    torch.save(img_tensor, img_save_path)
                if not os.path.exists(bboxes_save_path):
                    torch.save(bboxes_tensor, bboxes_save_path)
                
            elif os.path.exists(img_path) and os.path.exists(mask_path):

                img=io.imread(img_path)

                mask = io.imread(mask_path)
                
                original_size = img.shape
                img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)

                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
                mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) 

                img_save_path = os.path.join(target_path, 'img.pt')
                mask_save_path = os.path.join(target_path, 'mask.pt')
              
                if not os.path.exists(img_save_path):
                    torch.save(img_tensor, img_save_path)
                if not os.path.exists(mask_save_path):
                    torch.save(mask_tensor, mask_save_path)
