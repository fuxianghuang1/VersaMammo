import os
import torch
import numpy as np
import cv2
import pydicom as pdcm
from tqdm import tqdm
import torchvision.transforms.functional as F
from skimage import io

def preprocess(input_path,output_path,output_size=[224,224],dataset='CBIS-DDSM'):
    for folder_name in tqdm(os.listdir(input_path)):
        folder_path = os.path.join(input_path, folder_name)
        target_path=os.path.join(output_path, folder_name)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        if os.path.isdir(folder_path):
            if dataset=='CMMD':
                img1_path = os.path.join(folder_path, 'img1.jpg')
                img2_path = os.path.join(folder_path, 'img2.jpg')
                info_dict_path = os.path.join(folder_path, 'info_dict.npy')
                if os.path.exists(img1_path) and os.path.exists(img2_path) and os.path.exists(info_dict_path):
                    
                    img1=io.imread(img1_path)
                    img2=io.imread(img2_path)

                    info_dict = np.load(info_dict_path, allow_pickle=True).item()

                    info_dict = {k: str(v) for k, v in info_dict.items()}

                    img1 = cv2.resize(img1, output_size, interpolation=cv2.INTER_LINEAR)
                    img2 = cv2.resize(img2, output_size, interpolation=cv2.INTER_LINEAR)
   
                    img1_tensor = torch.tensor(img1, dtype=torch.float32).unsqueeze(0) # 添加通道维度
                    img2_tensor = torch.tensor(img2, dtype=torch.float32).unsqueeze(0)

          
                    img1_save_path = os.path.join(target_path, 'img1.pt')
                    img2_save_path = os.path.join(target_path, 'img2.pt')
                    info_dict_save_path = os.path.join(target_path, 'info_dict.pt')

                    torch.save(img1_tensor, img1_save_path)
                    torch.save(img2_tensor, img2_save_path)
                    torch.save(info_dict, info_dict_save_path)
            else:
                img_path = os.path.join(folder_path, 'img.jpg')
                info_dict_path = os.path.join(folder_path, 'info_dict.npy')

                if os.path.exists(img_path) and os.path.exists(info_dict_path):
                   
                    img=io.imread(img_path)

                    info_dict = np.load(info_dict_path, allow_pickle=True).item()
                    
                    original_size = img.shape
                    img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
  
                    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) # 添加通道维度

                    img_save_path = os.path.join(target_path, 'img.pt')
                    info_dict_save_path = os.path.join(target_path, 'info_dict.pt')

                    torch.save(img_tensor, img_save_path)
                    torch.save(info_dict, info_dict_save_path)
                
                