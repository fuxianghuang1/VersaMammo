import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from skimage import io

def preprocess(input_path,output_path,output_size=[224,224],dataset='CBIS-DDSM'):
    # 遍历输入路径下的所有文件夹
    for folder_name in tqdm(os.listdir(input_path)):
        folder_path = os.path.join(input_path, folder_name)
        target_path=os.path.join(output_path, folder_name)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        if os.path.isdir(folder_path):
           
            img_path = os.path.join(folder_path, 'img.jpg')
            info_dict_path = os.path.join(folder_path, 'info_dict.npy')

            if os.path.exists(img_path) and os.path.exists(info_dict_path):
                img=io.imread(img_path)

                info_dict = np.load(info_dict_path, allow_pickle=True).item()
                img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) # 添加通道维度

                img_save_path = os.path.join(target_path, 'img.pt')
                info_dict_save_path = os.path.join(target_path, 'info_dict.pt')

                torch.save(img_tensor, img_save_path)
                torch.save(info_dict, info_dict_save_path)
