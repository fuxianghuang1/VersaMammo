import argparse
import os
import pandas as pd 
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader  
from torchvision import transforms  
from breast_feature_extractor import BreastFeatureExtract
from datasets import DSDataset
import logging

logger = logging.getLogger("mammo") 


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_encoder_name", type=str, default='sigclip400m',                       
        choices=['mammoclip_b2','mammoclip_b5',
                 'medsam_vitb', 'lvmmed_vitb','lvmmed_r50','mama', 'versamammo'], 
        help="Select an image encoder.")
    parser.add_argument("--output_dir", type=str, default='/mnt/data/hfx/features/high')
    parser.add_argument("--data_dir", type=str, default='/mnt/data/hfx')
    parser.add_argument("--dataset", type=str, default='vindr',
        choices=['embed', 'rsna', 'vindr'],
                help="Select a dataset.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--resize_size", type=int, nargs=2, default=(512, 512),  
                        help="The size to which images should be resized. Format: width height") 
    parser.add_argument("--gpu", type=int, default=5, help="GPU index to use (default: 0)")  
    return parser

def main(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu') 
    model = BreastFeatureExtract(args).to(device)
    model.eval()
    resize_size = list(args.resize_size)
    transform  = transforms.Compose([  
                    transforms.Resize(resize_size),  
                    transforms.ToTensor(),  
                    transforms.Normalize(mean=[0.3089,0.3089, 0.3089], std=[0.2505, 0.2505, 0.2505])  
                ]) 
    print(model)
    print(transform)

    if args.dataset in ['embed', 'rsna', 'vindr']:
        df = pd.read_csv(f'csv_files/{args.dataset}.csv')

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}") 
    get_data = DSDataset    
    # df = df.sample(200)

    print('-------------extract training image features---------------')
    # df = df[df['split'] == "training"]
    dataset = get_data(df, transform=transform, data_dir = args.data_dir)  
    logger.info(f"# of dataset samples: {len(dataset):,d}")
    print(f"Loaded {len(dataset)} images.")
    print(f"Failed to load {len(dataset.get_failed_paths())} images.")  
    # print('the first data', dataset[0])

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, prefetch_factor=2)  # 你可以调整prefetch_factor的值
    all_features_and_paths = []

    with torch.no_grad():
        for data in tqdm(loader, desc="Extracting features"):
            try:
                images = data['image'].to(device)
                image_paths = data['image_path']  # 确保这是一个列表，或者根据需要进行转换

                # Extract features
                features = model(images)

                for i, feature in enumerate(features):
                    feature_np = feature.cpu().numpy()
                    all_features_and_paths.append({'feature': feature_np, 'image_path': image_paths[i]})
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue  # 或者你可以根据需要选择重新抛出异常或进行其他处理

    all_features_and_paths = pd.DataFrame(all_features_and_paths)
    save_dir = os.path.join(args.output_dir, args.dataset, args.image_encoder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_features_and_paths.to_pickle(os.path.join(save_dir, 'features.pkl'))

    print(f"Features of training set extracted and saved to {save_dir}")


if __name__ == "__main__":  
    args = get_args_parser().parse_args()  
    main(args)