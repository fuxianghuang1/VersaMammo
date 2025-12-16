import logging
import torch
from torch.utils.data import Dataset  
from PIL import Image  
from typing import Callable, Optional  
import pandas as pd  

logger = logging.getLogger("dinov2") 

class MammogramDataset(Dataset):  
    def __init__(self,  
                 image_paths_file: str,  
                 transform: Optional[Callable] = None,  
                 target_transform: Optional[Callable] = None) -> None:  
        self.transform = transform  
        self.target_transform = target_transform  
        df = pd.read_csv(image_paths_file) 
        # train_df = df[df['split'] == "training"].sample(n=1000)  
        train_df = df[df['split'] == "training"]

        self.image_paths = train_df['path'].tolist()  
        self.birads_labels = train_df['birads'].tolist()  
        self.density_labels = train_df['density'].tolist()  

        unique_birads_labels = train_df['birads'].unique()  
        unique_density_labels = train_df['density'].unique()  
        
        print("Unique BI-RADS labels:", unique_birads_labels)  
        print("Unique Density labels:", unique_density_labels)

        self.failed_paths = set()  
  
    def __len__(self):  
        return len(self.image_paths)  
  
    def __getitem__(self, index):  
        image_path = self.image_paths[index]  
  
        # image = None  
        try:  
            image = Image.open(image_path).convert('RGB')  
            image = image.resize((256, 256), Image.BILINEAR) 
            if self.transform:  
                image = self.transform(image)
                       
        except Exception as e:  
            logger.error(f"Error loading image at path {image_path}: {e}")  
            self.failed_paths.add(image_path)
            return None  

        if self.birads_labels[index] is None:

            birads_label = 0
        else:
            birads_label = self.birads_labels[index]


        if self.density_labels[index] is None:
            density_label = 0

        else:
            density_label = self.density_labels[index]
        return {
                'image': image,
                'birads_label': birads_label,
                'density_label': density_label
                }
  
    def get_failed_paths(self):  
        return list(self.failed_paths)  
