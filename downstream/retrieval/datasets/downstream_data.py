from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

class MultiViewDSDataset(Dataset):  
    def __init__(self, df, global_transform=None, local_transform=None, data_dir='/your/path/'):
        self.global_transform = global_transform if global_transform is not None else lambda x: x  
        self.local_transform = local_transform if local_transform is not None else lambda x: x 
        self.data_dir = data_dir

        self.CC_path = df['CC_path'].tolist()  
        self.MLO_path = df['MLO_path'].tolist() 

    def __len__(self):  
        return len(self.CC_path)  

    def __getitem__(self, index):  
        while index < len(self.CC_path):
            CC_path = os.path.join(self.data_dir, self.CC_path[index])
            MLO_path = os.path.join(self.data_dir, self.MLO_path[index])

            try:  
                if not os.path.exists(CC_path):
                    raise FileNotFoundError(f"CC image not found: {CC_path}")
                if not os.path.exists(MLO_path):
                    raise FileNotFoundError(f"MLO image not found: {MLO_path}")

                CC_img = Image.open(CC_path).convert('RGB')                  
                CC_img_global = self.global_transform(CC_img)
                CC_img_local = self.local_transform(CC_img)

                MLO_img = Image.open(MLO_path).convert('RGB') 
                MLO_img_global = self.global_transform(MLO_img)
                MLO_img_local = self.local_transform(MLO_img)

                return {
                    'CC_path': self.CC_path[index],
                    'MLO_path': self.MLO_path[index],
                    'CC_img_global': CC_img_global,
                    'CC_img_local': CC_img_local,
                    'MLO_img_global': MLO_img_global,
                    'MLO_img_local': MLO_img_local,
                }

            except Exception as e:  
                print(f"Error loading image: {e}")  

        raise IndexError("No valid images found.")

class DSDataset(Dataset):  
    def __init__(self, df, transform=None, data_dir='/your/path/'):
        self.transform = transform 
        self.data_dir = data_dir 
        self.image_paths = df['image_path'].tolist()  
        self.failed_paths = set()  
  
    def __len__(self):  
        return len(self.image_paths)  
  
    def __getitem__(self, index):  
        image_relative_path = self.image_paths[index] 
        image_path = os.path.join(self.data_dir, image_relative_path) 
  
        # image = None  
        try:  
            image = Image.open(image_path).convert('RGB')  
            # image = image.resize((256, 256), Image.BILINEAR) 
            if self.transform:  
                image = self.transform(image)
                       
        except Exception as e:  
            print(f"Error loading image at path {image_path}: {e}")  
            self.failed_paths.add(image_path)
            return None  
        
        return {
                'image': image,
                'image_path': image_relative_path,
                }
  
    def get_failed_paths(self):  
        return list(self.failed_paths)  

if __name__ == "__main__":
    df = pd.read_csv('csv_files/all_data_wo_dst5.csv')  
    print(df.columns)
    dataset = MultiViewDSDataset(df,  data_dir='/home/fuxiang/dataset')
    for i in range(5):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(sample)