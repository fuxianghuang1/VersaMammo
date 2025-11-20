import os
import time
import numpy as np
import time

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
from dataloader import myDataset,myNormalize
from torch.utils.data import DataLoader
from model import MultiTaskModel

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score
from sklearn.utils import resample
import contextlib
from sklearn.preprocessing import label_binarize
from preprocess import preprocess
from sklearn.preprocessing import OneHotEncoder
current_dir = os.path.dirname(os.path.abspath(__file__))


def valid(net, valid_dataloader, hypar, epoch=0):
    net.eval()
    # print("Validating...")
    print('----------'+hypar["restore_model"].split('/')[-1].split('.')[0]+'-----------')
    # epoch_num = hypar["max_epoch_num"]

    val_loss = 0.0
    val_cnt = 0.0
    tmp_time = []
    tmp_iou=[]

    total_iou = 0.0
    num_images = 0
    all_mean_ious = []

    start_valid = time.time()

    val_num = hypar['val_num']
    all_preds = {task: [] for task in net.classifiers.keys()}
    all_labels = {task: [] for task in net.classifiers.keys()}
    all_probs = {task: [] for task in net.classifiers.keys()}
    class_counts = {task: {label: 0 for label in label_mapping.keys()} for task, label_mapping in hypar['reverse_label_mappings'].items()}
    correct_counts = {task: {label: 0 for label in label_mapping.keys()} for task, label_mapping in hypar['reverse_label_mappings'].items()}

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
        
        batch_metrics = []
        for task, output in ds_val.items():
            preds = torch.argmax(output, dim=1)
            probs = torch.softmax(output, dim=1).detach() 
            all_preds[task].extend(preds.cpu().detach().numpy())
            all_labels[task].extend(labels_val_v[task].cpu().detach().numpy())
            all_probs[task].extend(probs.cpu().detach().numpy()) 

            for i, pred in enumerate(preds):
                true_label = int(labels_val_v[task][i].item())
                class_counts[task][true_label] += 1
                if true_label == pred:
                    correct_counts[task][true_label] += 1

            batch_bacc = balanced_accuracy_score(labels_val_v[task].cpu().numpy(), preds.cpu().numpy())
            batch_acc = accuracy_score(labels_val_v[task].cpu().numpy(), preds.cpu().numpy())
            batch_precision = precision_score(labels_val_v[task].cpu().numpy(), preds.cpu().numpy(), average='micro', zero_division=0)
            batch_recall = recall_score(labels_val_v[task].cpu().numpy(), preds.cpu().numpy(), average='micro')
            batch_f1 = f1_score(labels_val_v[task].cpu().numpy(), preds.cpu().numpy(), average='micro')

            batch_metrics.append((batch_bacc, batch_acc, batch_precision, batch_recall, batch_f1))
            # print(f'{i_val}/{val_num} for {task} - Acc: {batch_acc:.4f}, Precision: {batch_precision:.4f}, Recall: {batch_recall:.4f}, F1: {batch_f1:.4f}')

        gc.collect()
        torch.cuda.empty_cache()

    total_bacc, total_acc, total_precision, total_recall, total_f1, total_auc = 0, 0, 0, 0, 0, 0
    num_tasks = len(net.classifiers.keys())
    for task in net.classifiers.keys():
        all_preds[task] = np.array(all_preds[task])
        all_labels[task] = np.array(all_labels[task])
        all_probs[task] = np.array(all_probs[task])

        epoch_bacc = balanced_accuracy_score(all_labels[task], all_preds[task])
        epoch_acc = accuracy_score(all_labels[task], all_preds[task])
        epoch_precision = precision_score(all_labels[task], all_preds[task], average='macro', zero_division=0)
        epoch_recall = recall_score(all_labels[task], all_preds[task], average='macro')
        epoch_f1 = f1_score(all_labels[task], all_preds[task], average='macro')

        present_classes = np.unique(all_labels[task])
        y_true_epoch = label_binarize(all_labels[task], classes=present_classes)
        if y_true_epoch.shape[1]==1:
            
            encoder = OneHotEncoder(sparse_output=False)
            y_true_epoch = encoder.fit_transform(y_true_epoch)
        filtered_probs = all_probs[task][:, present_classes]
        epoch_auc = roc_auc_score(y_true_epoch, filtered_probs, average='macro')
        
        total_bacc += epoch_bacc
        total_acc += epoch_acc
        total_precision += epoch_precision
        total_recall += epoch_recall
        total_f1 += epoch_f1
        total_auc += epoch_auc

        print(f'Statistics for task {task}:')
        for label, count in class_counts[task].items():
            correct = correct_counts[task][label]
            print(f'  Label {hypar["reverse_label_mappings"][task][label]}: Total: {count}, Correct: {correct}')
    
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
        balanced_accuracy_bs = balanced_accuracy_score(y_true_bs, y_pred_bs)
        accuracy_bs = accuracy_score(y_true_bs, y_pred_bs)
        # balanced_acc_bs = balanced_accuracy_score(y_true_bs, y_pred_bs)
        f1_bs = f1_score(y_true_bs, y_pred_bs, average='macro')

        present_classes_bs = np.unique(y_true_bs)
        y_true_bin_bs = label_binarize(y_true_bs, classes=present_classes_bs)
        if y_true_bin_bs.shape[1] == 1:
            encoder_bs = OneHotEncoder(sparse_output=False)
            y_true_bin_bs = encoder_bs.fit_transform(y_true_bin_bs)
        filtered_probs_bs = y_prob_bs[:, present_classes_bs]
        auc_bs = roc_auc_score(y_true_bin_bs, filtered_probs_bs, average='macro')

        balanced_acc_scores.append(balanced_accuracy_bs)
        accuracy_scores.append(accuracy_bs)
        # balanced_acc_scores.append(balanced_acc_bs)
        f1_scores.append(f1_bs)
        auc_scores.append(auc_bs)

    # Calculate confidence intervals
    balanced_accuracy_ci = np.percentile(balanced_acc_scores, [2.5, 97.5])
    accuracy_ci = np.percentile(accuracy_scores, [2.5, 97.5])
    # balanced_acc_ci = np.percentile(balanced_acc_scores, [2.5, 97.5])
    f1_ci = np.percentile(f1_scores, [2.5, 97.5])
    auc_ci = np.percentile(auc_scores, [2.5, 97.5])
    
    ci_path=os.path.join(os.path.dirname(hypar["valid_out_dir"]),hypar["restore_model"].split('/')[-1].split('.')[0])
    if(not os.path.exists(ci_path)):
        os.makedirs(ci_path)
    np.save(os.path.join(ci_path,"bacc_ci.npy"),balanced_acc_scores)
    np.save(os.path.join(ci_path,"acc_ci.npy"),accuracy_scores)
    np.save(os.path.join(ci_path,"f1_ci.npy"),f1_scores)
    np.save(os.path.join(ci_path,"auc_ci.npy"),auc_scores)

    avg_bacc = total_bacc / num_tasks
    avg_acc = total_acc / num_tasks
    avg_precision = total_precision / num_tasks
    avg_recall = total_recall / num_tasks
    avg_f1 = total_f1 / num_tasks
    avg_auc = total_auc / num_tasks
    print(avg_auc)
    print('============================')
    print(f'Average metrics across all tasks - BAcc: {avg_bacc:.4f}, Acc: {avg_acc:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, AUC: {avg_auc:.4f}')
    print(f'Ci - BAcc_ci: [{balanced_accuracy_ci[0]:.4f}, {balanced_accuracy_ci[1]:.4f}], Acc_ci: [{accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f}], F1_ci: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}], Auc_ci: [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]')
    print('\n')
    return avg_acc, i_val, tmp_time

def main(hypar): 
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
    hypar['dataset']='INbreast'
    hypar['task']='Bi-Rads'
    hypar['finetune']='lp'#lp or ft
    hypar['input_path']=f'{current_dir}/../data/INbreast-classification'
    
    hypar['gpu_id']=[0]
    # hypar['base_path']=f'/home/jiayi/Baseline/classification/saved_model/{hypar["dataset"]}/{hypar["task"]}/{hypar["finetune"]}/'
    hypar["valid_out_dir"] = f"{current_dir}/results/{hypar['dataset']}/{hypar['task']}/{hypar['finetune']}"+'.txt'##"../DIS5K-Results-test" ## output inferenced segmentation maps into this fold
    hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
    hypar["seed"] = 0

    torch.manual_seed(hypar["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hypar["seed"])
        
    hypar["model_path"]=f"{current_dir}/saved_model/{hypar['dataset']}/{hypar['task']}/{hypar['finetune']}/"

    
    ## --- 2.5. define model  ---
    print("building model...")
    data_info = {
        'INbreast': {
            'Composition': {'Level A': 0, 'Level B': 1, 'Level C': 2, 'Level D': 3},
            'Bi-Rads': {'Bi-Rads 1': 0, 'Bi-Rads 2': 1, 'Bi-Rads 3': 2, 'Bi-Rads 4': 3,  'Bi-Rads 5': 4}
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
    
    hypar["batch_size_valid"] = 1 ## batch size for validation and inferencing
    
    info={hypar['task']:data_info[hypar['dataset']][hypar['task']]}
    
    hypar["input_size"] = [512, 512]
    hypar['val_datapath'] = hypar['input_path']+'/Test_cache_'+str(hypar["input_size"][0])
    if not os.path.exists(hypar['val_datapath']):
        preprocess(hypar['input_path']+'/Test',hypar['val_datapath'],hypar["input_size"])
    
    # # #EfficientNet-ours
    hypar["restore_model"]=hypar['model_path']+"VersaMammo"+".pth"
    hypar["model"]=MultiTaskModel('efficientnet', hypar['label_mappings'],checkpoint_path=f'{current_dir}/../../Sotas/VersaMammo/ENb5/ENB5_SL.pth',ours=None,finetune=hypar['finetune'])
    main(hypar=hypar)