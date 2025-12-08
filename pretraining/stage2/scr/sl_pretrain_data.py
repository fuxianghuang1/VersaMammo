from torch.utils.data import Dataset
from PIL import Image
import os  

class SupervisedPretrainDataset(Dataset):  
    def __init__(self, df, mode='train', transform1=None, transform2=None, train_num=None, data_dir='/your/new/path/'):
        self.mode = mode
        self.transform1 = transform1  
        self.transform2 = transform2
        self.data_dir = data_dir

        if self.mode == 'train':
            df = df[df['split'] == "training"]
            if train_num is not None:
                df = df.sample(train_num)
                print(f"Training samples used: {len(df)}")
        elif self.mode == 'test':
            df = df[df['split'] == "test"]
        elif self.mode == 'val':
            df = df[df['split'] == "val"]

        df = df.dropna(subset=['birads', 'density'])
        df = df[df['birads'] != 6]
        df = df[df['density'] != 5]
        df['density'] = df['density'].replace({1: 0, 2: 1, 3: 2, 4: 3})

        self.image_paths = df['image_path'].tolist()  
        self.birads_labels = df['birads'].tolist() 
        self.density_labels = df['density'].tolist() 

        unique_birads_labels = df['birads'].unique()  
        unique_density_labels = df['density'].unique()  

        print("Unique BI-RADS labels:", unique_birads_labels)  
        print("Unique Density labels:", unique_density_labels)

    def __len__(self):  
        return len(self.image_paths)  

    def __getitem__(self, index):  
        while index < len(self.image_paths):
            image_path = self.image_paths[index] 
            image_path = os.path.join(self.data_dir, image_path)

            try:  
                image = Image.open(image_path).convert('RGB')  

                if self.mode == 'train':
                    img = self.transform2(image)
                    return {
                        'img': img,
                        'birads_label': self.birads_labels[index],
                        'density_label': self.density_labels[index]
                    }
                else:
                    img = self.transform1(image)
                    return {
                        'img': img,
                        'birads_label': self.birads_labels[index],
                        'density_label': self.density_labels[index]
                    }

            except Exception as e:  
                print(f"Error loading image at path {image_path}: {e}")  
                index += 1  

        raise IndexError("No valid images found.")

class Stage2PretrainDataset(Dataset):  
    def __init__(self, df, mode='train', transform1=None, transform2=None, train_num= None, data_dir='/your/new/path/'):
        self.mode = mode
        self.transform1 = transform1  
        self.transform2 = transform2
        self.data_dir = data_dir

        if self.mode == 'train':
            df = df[df['split'] == "training"]
            if train_num is not None:
                df = df.sample(train_num) 
                print(f"Training samples used: {len(df)}")
        elif self.mode == 'test':
            df = df[df['split'] == "test"]
        elif self.mode == 'val':
            df = df[df['split'] == "val"]

        df = df.dropna(subset=['birads', 'density'])
        df = df[df['birads'] != 6]
        df = df[df['density'] != 5]
        df['density'] = df['density'].replace({1: 0, 2: 1, 3: 2, 4: 3})

        self.image_paths = df['image_path'].tolist()
        self.features = df['feature'].tolist()  
        self.birads_labels = df['birads'].tolist() 
        self.density_labels = df['density'].tolist() 

        unique_birads_labels = df['birads'].unique()  
        unique_density_labels = df['density'].unique()  

        print("Unique BI-RADS labels:", unique_birads_labels)  
        print("Unique Density labels:", unique_density_labels)

    def __len__(self):  
        return len(self.image_paths)  

    def __getitem__(self, index):  
        while index < len(self.image_paths):
            image_path = self.image_paths[index] 
            image_path = os.path.join(self.data_dir, image_path)

            try:  
                image = Image.open(image_path).convert('RGB')  

                if self.mode == 'train':
                    img = self.transform2(image)
                    return {
                        'img': img,
                        'birads_label': self.birads_labels[index],
                        'density_label': self.density_labels[index],
                        'global_feature': self.features[index]
                    }
                else:
                    img = self.transform1(image)
                    return {
                        'img': img,
                        'birads_label': self.birads_labels[index],
                        'density_label': self.density_labels[index]
                    }

            except Exception as e:  
                print(f"Error loading image at path {image_path}: {e}")  
                index += 1  

        raise IndexError("No valid images found.")

