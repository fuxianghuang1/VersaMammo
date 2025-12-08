import argparse
import os
import pandas as pd 
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader  
from torchvision import transforms  
from scr.breast_feature_extractor import BreastFeatureExtract
from scr.sl_pretrain_data import Stage2PretrainDataset 
# import logging
from sklearn.preprocessing import label_binarize
from sklearn.metrics import  accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.utils import resample
import time 
from scr.loss import contrastive_loss, kl_loss


# logger = logging.getLogger("mammo") 

class VersaMammo(nn.Module):
    def __init__(self, feature_extractor, args):
        super(VersaMammo, self).__init__()
        self.feature_extractor = feature_extractor
        # print (feature_extractor)

        self.birads_head = torch.nn.Linear(self.feature_extractor.out_dim, args.birads_n_class)
        self.density_head = torch.nn.Linear(self.feature_extractor.out_dim, args.density_n_class)

    def forward(self, x):
        features = self.feature_extractor(x)
        birads_out = self.birads_head(features)
        density_out = self.density_head(features)
        return birads_out, density_out, features 

def create_dataloader(df, mode='train', transform1=None, transform2=None, data_dir='/your/new/path/', batch_size=32):
    dataset = Stage2PretrainDataset(df, mode=mode, transform1=transform1, transform2=transform2, data_dir=data_dir)

    # 设置较小的num_workers和prefetch_factor
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=1,  # 降低CPU占用
        pin_memory=True,  # 使用pin_memory提高GPU加载效率
        prefetch_factor=batch_size,  # 控制预加载的样本数量
        drop_last=True
    )
    return dataloader 



def train_loop(model, dataloader, criterion_birads, criterion_density, optimizer, device, start_time, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    running_contrastive_loss = 0.0
    running_kl_loss = 0.0
    running_birads_ce_loss = 0.0
    running_density_ce_loss = 0.0
    total_batches = len(dataloader)

    for batch_index, batch in enumerate(dataloader):
        img = batch['img'].to(device)
        global_features = batch['global_feature'].to(device)
        birads_labels = batch['birads_label'].to(device).long()
        density_labels = batch['density_label'].to(device).long()

        birads_out, density_out, features = model(img)
        contrastive_loss_value = contrastive_loss(global_features, features, birads_labels, density_labels)   
        kl_loss_value = kl_loss(global_features, features) 
        birads_ce_loss_value = criterion_birads(birads_out, birads_labels) 
        density_ce_loss_value = criterion_density(density_out, density_labels) 

        total_loss = contrastive_loss_value + kl_loss_value +  (birads_ce_loss_value + density_ce_loss_value) 

        model.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 累加各个损失
        running_loss += total_loss.item()
        running_contrastive_loss += contrastive_loss_value.item()
        running_kl_loss += kl_loss_value.item()
        running_birads_ce_loss += birads_ce_loss_value.item()
        running_density_ce_loss += density_ce_loss_value.item()

        # 计算平均损失
        avg_loss = running_loss / (batch_index + 1)
        avg_contrastive_loss = running_contrastive_loss / (batch_index + 1)
        avg_kl_loss = running_kl_loss / (batch_index + 1)
        avg_birads_ce_loss = running_birads_ce_loss / (batch_index + 1)
        avg_density_ce_loss = running_density_ce_loss / (batch_index + 1)


        # # 计算并打印预计剩余时间
        elapsed_time = time.time() - start_time
        total_elapsed_time = elapsed_time + (num_epochs - epoch) * (total_batches * (elapsed_time / (batch_index + 1)))
        remaining_time = total_elapsed_time / (epoch - 1 + (batch_index + 1) / total_batches) - elapsed_time

        # time
        days = int(remaining_time // 86400)
        hours = int((remaining_time % 86400) // 3600)
        minutes = int((remaining_time % 3600) // 60)

        remaining_time_str = f"{days} days, {hours} hours, {minutes} minutes"

        # 打印每次迭代的损失
        print(f'Epoch [{epoch}/{num_epochs}], Batch [{batch_index + 1}/{total_batches}], '
              f'Avg loss: {avg_loss:.4f}, '
              f'Avg birads ce loss: {avg_birads_ce_loss:.4f}, '
              f'Avg density ce loss: {avg_density_ce_loss:.4f}, '
              f'Avg contrastive loss: {avg_contrastive_loss:.4f}, '
              f'Avg KL loss: {avg_kl_loss:.4f}, '
              f'Estimated Remaining Time: {remaining_time_str}')
            # 每个epoch保存三次模型
        if (epoch % 1 == 0 and batch_index == total_batches // 3 - 1) or \
        (epoch % 1 == 0 and batch_index == (total_batches // 3) * 2 - 1) or \
        (epoch % 1 == 0 and batch_index == total_batches - 1):
            model_path = os.path.join(args.output_dir, f'model_epoch_{epoch}_batch_{batch_index + 1}.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Model saved at epoch {epoch}, batch {batch_index + 1}')



def evaluate_model(y_true, y_pred, y_prob, n_bootstraps=1000, random_state=None):
    """
    Evaluate the model performance using accuracy, F1 score, balanced accuracy, and AUC with bootstrapped confidence intervals.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_prob: Predicted probabilities (shape: [n_samples, n_classes] for multi-class)

    Returns:
    - metrics: Dictionary containing F1 score, accuracy, balanced accuracy, AUC 
    """

    # Convert inputs to numpy arrays if they aren't already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    num_classes = len(np.unique(y_true))

    if num_classes == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        auc = roc_auc_score(y_true_bin, y_prob, average='macro')

    # Bootstrapping
    accuracy_scores = []
    balanced_acc_scores = []
    f1_scores = []
    auc_scores = []

    for _ in range(n_bootstraps):
        # Bootstrap sample
        indices = resample(np.arange(len(y_true)), random_state=random_state)
        y_true_bs = y_true[indices]
        y_pred_bs = y_pred[indices]
        y_prob_bs = y_prob[indices]

        # Calculate metrics for bootstrap sample
        accuracy_bs = accuracy_score(y_true_bs, y_pred_bs)
        balanced_acc_bs = balanced_accuracy_score(y_true_bs, y_pred_bs)
        f1_bs = f1_score(y_true_bs, y_pred_bs, average='macro')

        if num_classes == 2:
            auc_bs = roc_auc_score(y_true_bs, y_prob_bs[:, 1])
        else:
            y_true_bin_bs = label_binarize(y_true_bs, classes=np.unique(y_true))
            auc_bs = roc_auc_score(y_true_bin_bs, y_prob_bs, average='macro')

        accuracy_scores.append(accuracy_bs)
        balanced_acc_scores.append(balanced_acc_bs)
        f1_scores.append(f1_bs)
        auc_scores.append(auc_bs)

    # Calculate confidence intervals
    accuracy_ci = np.percentile(accuracy_scores, [2.5, 97.5])
    balanced_acc_ci = np.percentile(balanced_acc_scores, [2.5, 97.5])
    f1_ci = np.percentile(f1_scores, [2.5, 97.5])
    auc_ci = np.percentile(auc_scores, [2.5, 97.5])

    # Print results
    # print(f'Accuracy: {accuracy:.4f} (CI: {accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f})')
    # print(f'Balanced Accuracy: {balanced_acc:.4f} (CI: {balanced_acc_ci[0]:.4f}, {balanced_acc_ci[1]:.4f})')
    # print(f'F1 Score: {f1:.4f} (CI: {f1_ci[0]:.4f}, {f1_ci[1]:.4f})')
    # print(f'AUC: {auc:.4f} (CI: {auc_ci[0]:.4f}, {auc_ci[1]:.4f})')

    metrics = {
        'F1 Score': (f1, f1_ci),
        'Accuracy': (accuracy, accuracy_ci),
        'Balanced Accuracy': (balanced_acc, balanced_acc_ci),
        'AUC': (auc, auc_ci)
    }

    return metrics

def test_classifier(model, dataloader, device):
    model.eval()
    all_birads_labels = []
    all_density_labels = []
    all_predicted_birads = []
    all_predicted_density = []
    all_birads_probs = []
    all_density_probs = []

    with torch.no_grad():
        # for batch in tqdm(dataloader, desc="Testing"):
        for batch in dataloader:
            img = batch['img'].to(device)
            birads_labels = batch['birads_label']
            density_labels = batch['density_label']

            birads_out, density_out, _ = model(img)

            # Get predicted classes
            _, predicted_birads = torch.max(birads_out, 1)
            _, predicted_density = torch.max(density_out, 1)

            # Store labels and predictions
            all_birads_labels.extend(birads_labels.cpu().numpy())
            all_density_labels.extend(density_labels.cpu().numpy())
            all_predicted_birads.extend(predicted_birads.cpu().numpy())
            all_predicted_density.extend(predicted_density.cpu().numpy())

            # Store probabilities for AUC calculation
            all_birads_probs.extend(torch.softmax(birads_out, dim=1).cpu().numpy())
            all_density_probs.extend(torch.softmax(density_out, dim=1).cpu().numpy())

    # Calculate metrics
    # metric_birads = evaluate_model(birads_labels, predicted_birads, all_birads_probs)
    # metric_density = evaluate_model(density_labels, predicted_density, all_density_probs)
    # print ('metric_birads:', metric_birads)
    # print ('metric_density:', metric_density)

    accuracy_birads = (np.array(all_predicted_birads) == np.array(all_birads_labels)).sum() / len(all_birads_labels)
    accuracy_density = (np.array(all_predicted_density) == np.array(all_density_labels)).sum() / len(all_density_labels)

    balanced_acc_birads = balanced_accuracy_score(all_birads_labels, all_predicted_birads)
    balanced_acc_density = balanced_accuracy_score(all_density_labels, all_predicted_density)

    f1_birads = f1_score(all_birads_labels, all_predicted_birads, average='macro')
    f1_density = f1_score(all_density_labels, all_predicted_density, average='macro')

    # Calculate AUC for multi-class
    auc_birads = roc_auc_score(np.array(all_birads_labels), np.array(all_birads_probs), multi_class='ovr')
    auc_density = roc_auc_score(np.array(all_density_labels), np.array(all_density_probs), multi_class='ovr')

    eval_birads =  (balanced_acc_birads + f1_birads + auc_birads)/3
    eval_density = (balanced_acc_density + f1_density + auc_density)/3

    # Print metrics
    print(f"BI-RADS Accuracy: {accuracy_birads:.4f}, Balanced Accuracy: {balanced_acc_birads:.4f}, F1 Score: {f1_birads:.4f}, AUC: {auc_birads:.4f}")
    print(f"Density Accuracy: {accuracy_density:.4f}, Balanced Accuracy: {balanced_acc_density:.4f}, F1 Score: {f1_density:.4f}, AUC: {auc_density:.4f}")
    print(f"BI-RADS Avg: {eval_birads:.4f}")
    print(f"Density Avg: {eval_density:.4f}")

    # return eval_birads, eval_density

def mammo_ckpt_wrapper(ckpt):
    new_state_dict = {}
    for key, value in ckpt.items():
        if key.startswith('module.'):
            new_key = key[7:]  
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def main(args):
    args.birads_n_class = 7
    args.density_n_class = 4
    feature_extractor = BreastFeatureExtract(args)
    model = VersaMammo(feature_extractor,args)
    
    # Parse the GPU IDs
    gpu_ids = list(map(int, args.gpu_ids.split(',')))
    num_gpus = min(args.num_gpus, len(gpu_ids))  # Ensure we don't exceed available GPUs
    # Create necessary directories
    # args.output_dir = os.path.join(args.output_dir,args.image_encoder_name)
    os.makedirs(args.output_dir, exist_ok=True)
    # Set the device
    if num_gpus > 1:
        device = torch.device('cuda:{}'.format(gpu_ids[0]))  # Use the first GPU ID for the model
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)  # Wrap the model for multi-GPU
    else:
        device = torch.device('cuda:{}'.format(gpu_ids[0]))  # Use the specified single GPU ID

    model.to(device)

    resize_size = list(args.resize_size)

    transform1 = transforms.Compose([
        transforms.Resize(resize_size),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3089, 0.3089, 0.3089], std=[0.2505, 0.2505, 0.2505]) 
    ])
    transform2 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=resize_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3089, 0.3089, 0.3089], std=[0.2505, 0.2505, 0.2505]) 
    ])

    df = pd.read_pickle('files/vindr_breast_20k_w_teacher_features.pkl')  
    
    if args.test_only == 'yes':
        print("Testing mode activated.")
        # Load pre-trained parameters
        test_dataset = Stage2PretrainDataset(df, mode='test', transform1=transform1)
        print(f'Total testing samples: {len(test_dataset)}')

        # 使用create_dataloader创建DataLoader
        dataloader = create_dataloader(df, mode='test', transform1=transform1, data_dir=args.data_dir, batch_size=args.batch_size)

        # 获取所有保存的模型文件
        model_files = [f for f in os.listdir(args.output_dir) if f.endswith('.pth')]
        model_files.sort()  # 按名称排序，以确保按顺序加载

        for model_file in model_files:
            model_path = os.path.join(args.output_dir, model_file)
            print(f'Loading model from {model_path}...')
            if num_gpus == 1:
                checkpoint = mammo_ckpt_wrapper(torch.load(model_path, map_location='cpu'))
            else:
                checkpoint = torch.load(model_path, map_location='cpu')
            msg = model.load_state_dict(checkpoint, strict=False)
            print(msg)

            # 进行测试
            print(f'Testing model: {model_file}')
            test_classifier(model, dataloader, device)
    else:
        train_dataset = Stage2PretrainDataset(df, mode='train', transform1=transform1, transform2=transform2)
        print(f'Total training samples: {len(train_dataset)}')

        # 使用create_dataloader创建DataLoader
        dataloader = create_dataloader(df, mode='train', transform1=transform1, transform2=transform2, data_dir=args.data_dir, batch_size=args.batch_size)


        criterion_birads = torch.nn.CrossEntropyLoss()
        criterion_density = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in range(1, args.num_epochs + 1):  
            print(f"Epoch {epoch}/{args.num_epochs}")
            start_time = time.time()  # 开始时间
            train_loop(model, dataloader, criterion_birads, criterion_density, optimizer, device, start_time, epoch, args.num_epochs)
        print(f'Trainning {args.num_epochs} epoches: Models are saved in  {args.output_dir}')


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_encoder_name", type=str, default='enb5_in',                       
        choices=['enb5_in', 'enb5_rand'], 
        help="Select an image encoder.")
    parser.add_argument("--output_dir", type=str, default='versamammo_stage2_ckpt',)
    parser.add_argument("--data_dir", type=str, default='files')
    parser.add_argument("--batch_size", type=int, default=52)
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--resize_size", type=int, nargs=2, default=(512, 512),  
                        help="The size to which images should be resized. Format: width height") 
    parser.add_argument("--gpu_ids", type=str, default='0,1,2,3', help="Comma-separated list of GPU IDs to use (default: '1,2')")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use (default: 2)")
    parser.add_argument("--test_only", type=str, default='no', help="If set yes, only run testing.")
    return parser.parse_args()

if __name__ == "__main__":
    args = config()
    for k in args.__dict__.keys():
        print('    ', k, ':', str(args.__dict__[k]))
    main(args)

