import os
import time
import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

from dataloader import myDataset,myNormalize
from torch.utils.data import DataLoader
from model import MultiTaskModel
from preprocess import preprocess
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,roc_auc_score
from sklearn.utils import resample
import glob
import re
import contextlib
current_dir = os.path.dirname(os.path.abspath(__file__))

def get_path(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.pth"))

    # 正则表达式用于提取gpu_itr_和_traLoss之间的数字
    pattern = re.compile(r"gpu_itr_(\d+)_traLoss")

    # 找到数字最大的文件
    max_itr = -1
    latest_file = None

    for file in files:
        match = pattern.search(file)
        if match:
            itr = int(match.group(1))
            if itr > max_itr:
                max_itr = itr
                latest_file = file

    if latest_file:
        restore_model = latest_file
        print(f"The file with the largest gpu_itr is: {latest_file}")
    else:
        print("No matching files found.")
    return restore_model

def valid(net, valid_dataloader, hypar, epoch=0):
    net.eval()
    # print("Validating...")
    print('----------'+hypar["restore_model"].split('/')[-1].split('.')[0]+'-----------')
    epoch_num = hypar["max_epoch_num"]

    val_loss = 0.0
    val_cnt = 0.0
    tmp_time = []
    tmp_iou=[]

    total_iou = 0.0
    num_images = 0
    all_mean_ious = []

    start_valid = time.time()

    val_num = hypar['val_num']
    all_probs = {task: [] for task in hypar['label_mappings'].keys()}
    all_preds = {task: [] for task in hypar['label_mappings'].keys()}
    all_labels = {task: [] for task in hypar['label_mappings'].keys()}
    class_counts = {task: {label: 0 for label in hypar['label_mappings'][task].values()} for task in hypar['label_mappings'].keys()}
    correct_counts = {task: {label: 0 for label in hypar['label_mappings'][task].values()} for task in hypar['label_mappings'].keys()}

    for i_val, data_val in enumerate(valid_dataloader):
        val_cnt = val_cnt + 1.0
        imidx,image_name,inputs_val, labels_val = data_val['imidx'],data_val['image_name'],data_val['images'], data_val['labels']

        if(hypar["model_digit"]=="full"):
            inputs_val = inputs_val.type(torch.FloatTensor)
            labels_val = {task: labels.type(torch.long) for task, labels in labels_val.items()}
        # else:
            # inputs_val = inputs_val.type(torch.HalfTensor)
            # labels_val = {task: labels.type(torch.HalfTensor) for task, labels in labels_val.items()}

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_val_v = Variable(inputs_val.cuda(hypar['gpu_id'][0]), requires_grad=False)
            labels_val_v = {task: Variable(labels.cuda(hypar['gpu_id'][0]), requires_grad=False) for task, labels in labels_val.items()}
        else:
            inputs_val_v = Variable(inputs_val, requires_grad=False)
            labels_val_v = {task: Variable(labels, requires_grad=False) for task, labels in labels_val.items()}

        t_start = time.time()
        ds_val = net(inputs_val_v)
        

        probs = {task: ds_val[task].detach() for task in ds_val}
        preds = {task: (ds_val[task] > 0.5).float() for task in ds_val}

        # all_preds = {task: [] for task in ds_val}
        # all_labels = {task: [] for task in ds_val}
        # class_counts = {task: [0] * len(hypar['label_mappings'][task]) for task in ds_val}
        # correct_counts = {task: [0] * len(hypar['label_mappings'][task]) for task in ds_val}

        for task in preds:
            all_probs[task].append(probs[task].cpu().numpy())
            all_preds[task].append(preds[task].cpu().numpy())
            all_labels[task].append(labels_val_v[task].cpu().numpy())
            # Update class counts and correct counts
            for i in range(labels_val_v[task].size(0)):
                indices = torch.nonzero(labels_val_v[task][i] == 1)[-1].cpu().tolist()
                for key in indices:
                    class_counts[task][key]+=1
                
                for key in indices:
                    if preds[task][i][key]==labels_val_v[task][i][key]:
                        correct_counts[task][key] += 1

    # Convert lists to numpy arrays
    for task in all_preds:
        all_probs[task] = np.vstack(all_probs[task])
        all_preds[task] = np.vstack(all_preds[task])
        all_labels[task] = np.vstack(all_labels[task])
    # print(all_labels[task])
    # Calculate overall metrics for each task
    total_acc, total_precision, total_recall, total_f1, total_auc = 0, 0, 0, 0, 0
    num_tasks = len(hypar['label_mappings'])

    for task in all_preds:
        epoch_acc = accuracy_score(all_labels[task], all_preds[task])
        epoch_precision = precision_score(all_labels[task], all_preds[task], average='macro', zero_division=0)
        epoch_recall = recall_score(all_labels[task], all_preds[task], average='macro')
        epoch_f1 = f1_score(all_labels[task], all_preds[task], average='macro')

        non_zero_indices = np.sum(all_labels[task], axis=0) != 0
        epoch_auc = roc_auc_score(all_labels[task][:,non_zero_indices], all_probs[task][:,non_zero_indices], average='macro')

        total_acc += epoch_acc
        total_precision += epoch_precision
        total_recall += epoch_recall
        total_f1 += epoch_f1
        total_auc += epoch_auc

        print(f'Statistics for task {task}:')
        reverse_label_mapping = {v: k for k, v in hypar['label_mappings'][task].items()}
        for label, count in class_counts[task].items():
            correct = correct_counts[task][label]
            label_name = reverse_label_mapping[label]
            print(f'  Label {label_name}: Total: {count}, Correct: {correct}')
    accuracy_scores = []
    balanced_acc_scores = []
    f1_scores = []
    auc_scores = []
    y_true = np.concatenate([all_labels[task] for task in net.classifiers.keys()])
    y_pred = np.concatenate([all_preds[task] for task in net.classifiers.keys()])
    y_prob = np.concatenate([all_probs[task] for task in net.classifiers.keys()])
    num_classes = len(np.unique(y_true))  

    for _ in range(1000):
        # Bootstrap sample
        indices = resample(np.arange(len(y_true)), random_state=None)
        y_true_bs = y_true[indices]
        y_pred_bs = y_pred[indices]
        y_prob_bs = y_prob[indices]

        # Calculate metrics for bootstrap sample
        accuracy_bs = accuracy_score(y_true_bs, y_pred_bs)
        # balanced_acc_bs = balanced_accuracy_score(y_true_bs, y_pred_bs)
        f1_bs = f1_score(y_true_bs, y_pred_bs, average='macro')

        non_zero_indices_bs = np.sum(y_true_bs, axis=0)!= 0
        if np.sum(non_zero_indices_bs) == 0:
            auc_bs = 0
        else:
            auc_bs = roc_auc_score(y_true_bs[:, non_zero_indices_bs], y_prob_bs[:, non_zero_indices_bs], average='macro')
        accuracy_scores.append(accuracy_bs)
        # balanced_acc_scores.append(balanced_acc_bs)
        f1_scores.append(f1_bs)
        auc_scores.append(auc_bs)

    # Calculate confidence intervals
    accuracy_ci = np.percentile(accuracy_scores, [2.5, 97.5])
    # balanced_acc_ci = np.percentile(balanced_acc_scores, [2.5, 97.5])
    f1_ci = np.percentile(f1_scores, [2.5, 97.5])
    auc_ci = np.percentile(auc_scores, [2.5, 97.5])
    ci_path=os.path.join(os.path.dirname(hypar["valid_out_dir"]),hypar["restore_model"].split('/')[-1].split('.')[0])
    if(not os.path.exists(ci_path)):
        os.makedirs(ci_path)
    np.save(os.path.join(ci_path,"acc_ci.npy"),accuracy_scores)
    np.save(os.path.join(ci_path,"f1_ci.npy"),f1_scores)
    np.save(os.path.join(ci_path,"auc_ci.npy"),auc_scores)

    avg_acc = total_acc / num_tasks
    avg_precision = total_precision / num_tasks
    avg_recall = total_recall / num_tasks
    avg_f1 = total_f1 / num_tasks
    avg_auc = total_auc / num_tasks
    print(avg_auc)
    print('============================')
    print(f'Average metrics across all tasks - Acc: {avg_acc:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, AUC: {avg_auc:.4f}')
    print(f'Ci - Acc_ci: [{accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f}], F1_ci: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}], Auc_ci: [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]')
    print('\n')

    return avg_acc, i_val, tmp_time


def main(hypar): # model: "train", "test"
    val_dataset=myDataset(hypar['val_datapath'],[myNormalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])],hypar['label_mappings'])
    hypar['val_num']=len(val_dataset)
    val_dataloader=DataLoader(val_dataset, batch_size=hypar["batch_size_valid"], shuffle=False, num_workers=1, pin_memory=False)

    print("--- build model ---")
    net = hypar["model"]

    if(hypar["model_digit"]=="half"):
        net.half()
        for layer in net.modules():
          if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    if torch.cuda.is_available():
        if len(hypar['gpu_id']) > 1:
            net = net.cuda(hypar['gpu_id'][0])
            net = nn.DataParallel(net, device_ids=hypar['gpu_id'])
        else:
            net = net.cuda(hypar['gpu_id'][0])

    if(hypar["restore_model"]!=""):
        print("restore model from:")
        print(hypar["restore_model"])
        if torch.cuda.is_available():
            if len(hypar['gpu_id']) > 1:
                net.load_state_dict(torch.load(hypar["restore_model"], map_location=lambda storage, loc: storage.cuda(hypar['gpu_id'][0])))
            else:
                # model = model.cuda(hypar['gpu_id'][0])
                pretrained_dict = torch.load(hypar["restore_model"], map_location=lambda storage, loc: storage.cuda(hypar['gpu_id'][0]))
                pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
                net.load_state_dict(pretrained_dict)
        else:
            net.load_state_dict(torch.load(hypar["restore_model"], map_location='cpu'))

    if not os.path.exists(os.path.dirname(hypar['valid_out_dir'])):
        os.makedirs(os.path.dirname(hypar['valid_out_dir']))
    with open(hypar['valid_out_dir'], 'a') as f:
        with contextlib.redirect_stdout(f):
            valid(net, val_dataloader, hypar)


if __name__ == "__main__":
    hypar = {}
    hypar["mode"] = "eval"
    hypar['dataset']='DMID-finding'
    hypar['task']='Finding'
    hypar['input_path']=f'../../datapre/classification_data/{hypar["dataset"]}'
    hypar['finetune']='lp'#lp or ft
    hypar["valid_out_dir"] = f"{current_dir}/results/{hypar['dataset']}/{hypar['task']}/{hypar['finetune']}"+'.txt'##"../DIS5K-Results-test" ## output inferenced segmentation maps into this fold
    hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
    hypar["seed"] = 0
    hypar['gpu_id']=[1]

    torch.manual_seed(hypar["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hypar["seed"])

    
    print("building model...")
    data_info = {
        'VinDr-Mammo-finding': {
            'Finding': {
                'Mass': 0,
                'Global Asymmetry': 1,
                'Architectural Distortion': 2,
                'Nipple Retraction': 3,
                'Suspicious Calcification': 4,
                'Focal Asymmetry': 5,
                'Asymmetry': 6,
                'Suspicious Lymph Node': 7,
                'Skin Thickening': 8,
                'Skin Retraction': 9
            }
        },
        'CMMD':{
            'Finding':{
                'Mass':0,
                'Calcification':1
            }
        },
        'INbreast':{
            'Finding':{
                'Mass':0,
                'Calcification':1,
                'Asymmetry':2,
                'Architectural distortion':3,
                'Normal':4
            }
        },
        'DMID-finding':{
            'Finding':{
                'Circumscribed masses':0,
                'Calcification':1,
                'Asymmetry':2,
                'Architectural distortion':3,
                'Spiculated masses':4,
                'Miscellaneous':5
            }
        },

    }

    hypar['label_mappings'] = {hypar['task']:data_info[hypar['dataset']][hypar['task']]}
    def create_reverse_label_mapping(data_info):
        reverse_label_mapping = {}
        
        for dataset, mappings in data_info.items():
            reverse_label_mapping[dataset] = {}
            for category, mapping in mappings.items():
                reverse_label_mapping[dataset][category] = {v: k for k, v in mapping.items()}
        
        return reverse_label_mapping
    hypar['reverse_label_mappings'] = {hypar['task']:create_reverse_label_mapping(data_info)[hypar['dataset']][hypar['task']]}
    # hypar["model"] = ViTB_Decoder(1,hypar['checkpoint_path']) #U2NETFASTFEATURESUP()
    """['resnet50', 'efficientnet_b2', 'vit_base_patch16_224']"""
    """['vitb14imagenet','vitb14dinov2mammo','vitb14dinov2mammo1']"""
    # hypar["model"]=MultiTaskModel('vit_base_patch16_224', data_info[hypar['dataset']], pretrained=True,checkpoint_path=None,ours='vitb14dinov2mammo1')
    hypar["early_stop"] = 20 ## stop the training when no improvement in the past 20 validation periods, smaller numbers can be used here e.g., 5 or 10.
    hypar["model_save_fre"] = 500 ## valid and save model weights every 2000 iterations

    hypar["batch_size_train"] = 16 ## batch size for training
    hypar["batch_size_valid"] = 1 ## batch size for validation and inferencing
    # print("batch size: ", hypar["batch_size_train"])

    hypar["max_ite"] = 10000000 ## if early stop couldn't stop the training process, stop it by the max_ite_num
    hypar["max_epoch_num"] = 1000000 ## if early stop and max_ite couldn't stop the training process, stop it by the max_epoch_num
    

    info={hypar['task']:data_info[hypar['dataset']][hypar['task']]}
    
    hypar["input_size"] = [512, 512] 
    hypar['val_datapath'] = hypar['input_path']+'/Test_cache_'+str(hypar["input_size"][0])
    if not os.path.exists(hypar['val_datapath']):
        preprocess(hypar['input_path']+'/Test',hypar['val_datapath'],hypar["input_size"])
        
    hypar["model_path"]=f"{current_dir}/saved_model_512_cnn_auc/{hypar['dataset']}/{hypar['task']}/{hypar['finetune']}/"
    
    # # # # #resnet50-lvmmed
    hypar["restore_model"]=hypar['model_path']+"LVM-Med (R50)"+".pth"
    hypar["model"]=MultiTaskModel('resnet50', info,checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/lvmmed/lvmmed_resnet.torch',ours=None)
    main(hypar=hypar)
    
    # # #vitb-lvmmed
    hypar["restore_model"]=hypar['model_path']+"LVM-Med (Vitb)"+".pth" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('vit_base_patch16_224', info,checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/lvmmed/lvmmed_vit.pth',ours=None)
    main(hypar=hypar)
    
    # # #vitb-medsam
    hypar["restore_model"]=hypar['model_path']+"MedSAM (Vitb)"+".pth" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('vit_base_patch16_224', info,checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/medsam_vit_b.pth',ours=None)
    main(hypar=hypar)
    
    # #mammo-clip-b2
    hypar["restore_model"]=hypar['model_path']+"Mammo-CLIP (Enb2)"+".pth"
    hypar["model"]=MultiTaskModel('efficientnet', info,checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/mammo-clip/b2-model-best-epoch-10.tar',ours=None)
    main(hypar=hypar)
    
    # # #mammo-clip-b5
    hypar["restore_model"]=hypar['model_path']+"Mammo-CLIP (Enb5)"+".pth" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('efficientnet', info,checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/mammo-clip/b5-model-best-epoch-7.tar',ours=None)
    main(hypar=hypar)
    
    # # #EfficientNet-ours
    hypar["restore_model"]=hypar['model_path']+"VersaMammo"+".pth"
    hypar["model"]=MultiTaskModel('efficientnet', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/VersaMammo/ENb5/ENB5_SL.pth',ours=None)
    main(hypar=hypar)
    
    hypar["input_size"] = [518, 518]
    hypar['val_datapath'] = hypar['input_path']+'/Test_cache_'+str(hypar["input_size"][0])
    if not os.path.exists(hypar['val_datapath']):
        preprocess(hypar['input_path']+'/Test',hypar['val_datapath'],hypar["input_size"])
    
    # # # #MAMA
    hypar["restore_model"]=hypar['model_path']+"MAMA (Vitb)"+".pth"
    hypar["model"]=MultiTaskModel('MAMA', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/mama_embed_pretrained_40k_steps_last.ckpt',ours=None)
    main(hypar=hypar)
