import logging
from torch.utils.data import Dataset  
from PIL import Image  
import os

logger = logging.getLogger("mammo") 

class RsnaDataset(Dataset):  
    def __init__(self, df, transform=None):
        self.transform = transform  

        self.image_paths = df['path'].tolist()  
        self.birads_labels = df['birads'].tolist() 
        self.density_labels = df['density'].tolist() 
        self.invasive_labels = df['invasive'].tolist() 
        self.cancer_labels = df['cancer'].tolist() 

        unique_birads_labels = df['birads'].unique()  
        unique_density_labels = df['density'].unique() 
        unique_invasive_labels = df['invasive'].unique()  
        unique_cancer_labels = df['cancer'].unique()   
        
        print("Unique BI-RADS labels:", unique_birads_labels)  
        print("Unique Density labels:", unique_density_labels)
        print("Unique Invasive labels:", unique_invasive_labels)  
        print("Unique Cancer labels:", unique_cancer_labels)


        self.failed_paths = set()  
  
    def __len__(self):  
        return len(self.image_paths)  
  
    def __getitem__(self, index):  
        image_relative_path = self.image_paths[index] 
        image_path = os.path.join('/mnt/data/hfx', image_relative_path)  
  
        # image = None  
        try:  
            image = Image.open(image_path).convert('RGB')  
            # image = image.resize((256, 256), Image.BILINEAR) 
            if self.transform:  
                image = self.transform(image)
                       
        except Exception as e:  
            logger.error(f"Error loading image at path {image_path}: {e}")  
            self.failed_paths.add(image_path)
            return None  

        birads_label = self.birads_labels[index]
        density_label = self.density_labels[index]
        invasive_label = self.invasive_labels[index]
        cancer_label = self.cancer_labels[index]
        
        return {
                'image': image,
                'image_path': image_relative_path,
                'birads_label': birads_label,
                'density_label': density_label,
                'invasive_label': invasive_label,
                'cancer_label': cancer_label
                }
  
    def get_failed_paths(self):  
        return list(self.failed_paths)  