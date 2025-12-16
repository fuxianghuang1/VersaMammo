from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

class PretrainDataset(Dataset):  
    def __init__(self, df, mode='train', global_transform=None, local_transform=None, data_dir='/your/newd/path/'):
        self.mode = mode
        self.global_transform = global_transform if global_transform is not None else lambda x: x  
        self.local_transform = local_transform if local_transform is not None else lambda x: x 
        self.data_dir = data_dir

        if self.mode == 'train':
            df = df[df['split'] == "train"]
        elif self.mode == 'test':
            df = df[df['split'] == "test"]
            df = df.dropna(subset=['birads', 'density'])
        elif self.mode == 'val':
            df = df[df['split'] == "val"]
            df = df.dropna(subset=['birads', 'density'])

        
        df['birads'] = df['birads'].fillna(-1)  # nan : -1
        df['density'] = df['density'].fillna(-1)

        self.CC_path = df['CC_path'].tolist()  
        self.MLO_path = df['MLO_path'].tolist() 
        self.birads_labels = df['birads'].tolist() 
        self.density_labels = df['density'].tolist() 

        # 检查标签长度
        if not (len(self.CC_path) == len(self.MLO_path) == len(self.birads_labels) == len(self.density_labels)):
            raise ValueError("Length of paths and labels must be the same.")

        unique_birads_labels = df['birads'].unique()  
        unique_density_labels = df['density'].unique()  

        print("Unique BI-RADS labels:", unique_birads_labels)  
        print("Unique Density labels:", unique_density_labels)

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
                    'CC_img_global': CC_img_global,
                    'CC_img_local': CC_img_local,
                    'MLO_img_global': MLO_img_global,
                    'MLO_img_local': MLO_img_local,
                    'birads_label': self.birads_labels[index],
                    'density_label': self.density_labels[index]
                }

            except Exception as e:  
                print(f"Error loading image: {e}")  

        raise IndexError("No valid images found.")

if __name__ == "__main__":
    df = pd.read_csv('csv_files/all_data_wo_dst5.csv')  
    print(df.columns)
    dataset = PretrainDataset(df, mode='train', data_dir='/home/fuxiang/dataset')
    for i in range(5):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(sample)