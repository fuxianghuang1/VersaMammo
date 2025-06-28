import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import contextlib

def compute_metrics(pred, gt):
    pred = (pred > 127).float()
    gt = (gt > 127).float()

    tp = torch.sum((pred == 1) & (gt == 1)).float()
    fp = torch.sum((pred == 1) & (gt == 0)).float()
    tn = torch.sum((pred == 0) & (gt == 0)).float()
    fn = torch.sum((pred == 0) & (gt == 1)).float()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else torch.tensor(1.0)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else torch.tensor(1.0)
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else torch.tensor(1.0)
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else torch.tensor(1.0)
    pred_area = torch.sum(pred).float()
    gt_area = torch.sum(gt).float()
    relative_area_diff = torch.abs(pred_area - gt_area) / gt_area if gt_area != 0 else torch.tensor(0.0)

    return accuracy, sensitivity, specificity, jaccard, dice, relative_area_diff

def load_image(path):
    img = Image.open(path).convert('L')
    img = torch.tensor(np.array(img), dtype=torch.float32)
    return img

def main(pred_folder, gt_folder, log_file,model):
    pred_files = sorted(os.listdir(pred_folder))
    metrics = {
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'jaccard': [],
        'dice': [],
        'relative_area_diff': []
    }

    with open(log_file, 'a') as f:
        with contextlib.redirect_stdout(f):
            for pred_file in tqdm(pred_files):
                pred_path = os.path.join(pred_folder, pred_file)
                gt_path = os.path.join(gt_folder, pred_file.split('.')[0], 'mask.png')

                pred = load_image(pred_path)
                gt = load_image(gt_path)
                
                acc, sen, spe, jac, dic, rad = compute_metrics(pred, gt)
                
                metrics['accuracy'].append(acc.item())
                metrics['sensitivity'].append(sen.item())
                metrics['specificity'].append(spe.item())
                metrics['jaccard'].append(jac.item())
                metrics['dice'].append(dic.item())
                metrics['relative_area_diff'].append(rad.item())

            for key in metrics:
                metrics[key] = np.mean(metrics[key])
            print(f"-----------------{model}-------------------")
            print(f"Average Accuracy: {metrics['accuracy']:.4f}")
            print(f"Average Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"Average Specificity: {metrics['specificity']:.4f}")
            print(f"Average Jaccard Index: {metrics['jaccard']:.4f}")
            print(f"Average Dice Score: {metrics['dice']:.4f}")
            print(f"Average Relative Area Difference: {metrics['relative_area_diff']:.4f}")
            print('\n')

# 设置预测图像和真实图像文件夹路径
base_path='/home/jiayi/Baseline/Segment/results/INbreast-cropped'
gt_folder = '/home/jiayi/Baseline/segdetdata/INbreast-cropped/Test'
log_file = '/home/jiayi/Baseline/Segment/txt_result/INbreast-cropped.txt'  # 替换为你希望保存日志的路径
for model in os.listdir(base_path):
    pred_folder = os.path.join(base_path,model)
    # 计算并打印平均指标，并将输出写入日志文件
    main(pred_folder, gt_folder, log_file,model)
