import os
import time
import numpy as np
from skimage import io
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch, gc
import torch.nn as nn
from Mammo_VQA_dataset import MammoVQA_image,custom_collate_fn
from torch.utils.data import DataLoader
from model import MultiTaskModel

from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score
import contextlib
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    all_preds = {net.question_topic: []}
    all_labels = {net.question_topic: []}
    all_probs = {net.question_topic: []}
    class_counts = {question_topic: {label: 0 for label in label_mapping.keys()} for question_topic, label_mapping in hypar['reverse_label_mappings'].items()}
    correct_counts = {question_topic: {label: 0 for label in label_mapping.keys()} for question_topic, label_mapping in hypar['reverse_label_mappings'].items()}

    for i_val, data_val in enumerate(valid_dataloader):
        val_cnt = val_cnt + 1.0
        image, question, label, idx = data_val

        t_start = time.time()
        logits, loss= net(image, question, label,hypar["input_size"])
        label=torch.tensor(label)
        batch_metrics = []
        for task, output in logits.items():
            preds = torch.argmax(output, dim=1)
            probs = torch.softmax(output, dim=1).detach()  
            all_preds[task].extend(preds.cpu().detach().numpy())
            all_labels[task].extend(label.cpu().detach().numpy())
            all_probs[task].extend(probs.cpu().detach().numpy())  
            for i, pred in enumerate(preds):
                true_label = int(label[i].item())
                class_counts[task][true_label] += 1
                if true_label == pred:
                    correct_counts[task][true_label] += 1
            batch_bacc = balanced_accuracy_score(label.cpu().numpy(), preds.cpu().numpy())
            batch_acc = accuracy_score(label.cpu().numpy(), preds.cpu().numpy())
            batch_precision = precision_score(label.cpu().numpy(), preds.cpu().numpy(), average='micro', zero_division=0)
            batch_recall = recall_score(label.cpu().numpy(), preds.cpu().numpy(), average='micro')
            batch_f1 = f1_score(label.cpu().numpy(), preds.cpu().numpy(), average='micro')

            batch_metrics.append((batch_bacc, batch_acc, batch_precision, batch_recall, batch_f1))
            # print(f'{i_val}/{val_num} for {task} - Acc: {batch_acc:.4f}, Precision: {batch_precision:.4f}, Recall: {batch_recall:.4f}, F1: {batch_f1:.4f}')

        gc.collect()
        torch.cuda.empty_cache()

    total_bacc, total_acc, total_precision, total_recall, total_f1, total_auc = 0, 0, 0, 0, 0, 0
    num_tasks = 1
    task=net.question_topic
    # for task in net.classifiers.keys():
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

    y_true = all_labels[task]
    y_pred = all_preds[task]
    y_prob = all_probs[task]
    num_classes = len(np.unique(y_true))  

    for _ in range(1000):
        # Bootstrap sample
        indices = resample(np.arange(len(y_true)), random_state=None)
        y_true_bs = y_true[indices]
        y_pred_bs = y_pred[indices]
        y_prob_bs = y_prob[indices]

        # Calculate metrics for bootstrap sample
        accuracy_bs = accuracy_score(y_true_bs, y_pred_bs)
        balanced_acc_bs = balanced_accuracy_score(y_true_bs, y_pred_bs)
        f1_bs = f1_score(y_true_bs, y_pred_bs, average='macro')

        present_classes_bs = np.unique(y_true_bs)
        y_true_bin_bs = label_binarize(y_true_bs, classes=present_classes_bs)
        if y_true_bin_bs.shape[1] == 1:
            encoder_bs = OneHotEncoder(sparse_output=False)
            y_true_bin_bs = encoder_bs.fit_transform(y_true_bin_bs)
        filtered_probs_bs = y_prob_bs[:, present_classes_bs]
        auc_bs = roc_auc_score(y_true_bin_bs, filtered_probs_bs, average='macro')

        accuracy_scores.append(accuracy_bs)
        balanced_acc_scores.append(balanced_acc_bs)
        f1_scores.append(f1_bs)
        auc_scores.append(auc_bs)

    # Calculate confidence intervals
    accuracy_ci = np.percentile(accuracy_scores, [2.5, 97.5])
    balanced_acc_ci = np.percentile(balanced_acc_scores, [2.5, 97.5])
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

    print('============================')
    print(f'Average metrics across all tasks - BAcc: {avg_bacc:.4f}, Acc: {avg_acc:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, AUC: {avg_auc:.4f}')
    print(f'Ci - BAcc_ci: [{balanced_acc_ci[0]:.4f}, {balanced_acc_ci[1]:.4f}], Acc_ci: [{accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f}], F1_ci: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}], Auc_ci: [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]')
    print('\n')
    return avg_acc, i_val, tmp_time

def main(hypar): # model: "train", "test"

    val_dataset=MammoVQA_image(hypar['val_data'],hypar['label_mappings'])
    hypar['val_num']=len(val_dataset)
    val_dataloader=DataLoader(val_dataset, collate_fn=custom_collate_fn, batch_size=hypar["batch_size_valid"], shuffle=False, num_workers=1, pin_memory=False)

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
                net.load_state_dict(pretrained_dict, strict=False)
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
    hypar['question topic']='Bi-Rads'
    hypar['finetune']='lp'#lp or ft
    hypar["valid_out_dir"] = f"{os.path.dirname(current_dir)}/VQA/results/{hypar['question topic']}/{hypar['finetune']}"+'.txt'##"../DIS5K-Results-test" ## output inferenced segmentation maps into this fold
    hypar['gpu_id']=[0]
    hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
    hypar["seed"] = 0
    hypar["start_ite"]=0

    torch.manual_seed(hypar["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hypar["seed"])

    hypar['val_datapath'] = f'{current_dir}/MammoVQA-Image-Bench.json'
    with open(hypar['val_datapath'], 'r') as f:
        val_data = json.load(f)
    hypar['val_data'] = [{"ID": key, **value} for key, value in val_data.items() if value["Question topic"] == hypar['question topic']]
    hypar["model_path"]=f"{current_dir}/saved_model/{hypar['question topic']}/{hypar['finetune']}/"

    data_info = {
        'View': {'MLO':0,'CC':1},
        'Laterality': {'Right':0,'Left':1},
        'Pathology': {'Normal':0,'Malignant':1,'Benign':2},
        'ACR': {'Level A':0,'Level B':1,'Level C':2,'Level D':3},
        'Subtlety': {'Normal':0,'Level 1':1,'Level 2':2,'Level 3':3,'Level 4':4,'Level 5':5},
        'Bi-Rads': {'Bi-Rads 0':0,'Bi-Rads 1':1,'Bi-Rads 2':2,'Bi-Rads 3':3,'Bi-Rads 4':4,'Bi-Rads 5':5},
        'Masking potential': {'Level 1':0,'Level 2':1,'Level 3':2,'Level 4':3,'Level 5':4,'Level 6':5,'Level 7':6,'Level 8':7},
    }


    hypar['label_mappings'] = {hypar['question topic']:data_info[hypar['question topic']]}
    def create_reverse_label_mapping(data_info):
        """
        Creates a reverse mapping from index to label for each category in data_info.

        Args:
            data_info (dict): A dictionary containing label mappings for various categories.

        Returns:
            dict: A dictionary containing reverse mappings for each category.
        """
        reverse_label_mapping = {}

        for category, mapping in data_info.items():
            # Reverse the label mapping for the current category
            reverse_label_mapping[category] = {v: k for k, v in mapping.items()}

        return reverse_label_mapping

    hypar['reverse_label_mappings'] = {hypar['question topic']:create_reverse_label_mapping(data_info)[hypar['question topic']]}
    # hypar["model"] = ViTB_Decoder(1,hypar['checkpoint_path']) #U2NETFASTFEATURESUP()
    """['resnet50', 'efficientnet_b2', 'vit_base_patch16_224']"""
    """['vitb14imagenet','vitb14dinov2mammo','vitb14dinov2mammo1']"""

    hypar["batch_size_valid"] = 1 ## batch size for validation and inferencing

    hypar["max_ite"] = 10000000 
    
    hypar["input_size"] = [512, 512]
    
    # # # #resnet50-lvmmed
    hypar['restore_model'] =hypar['model_path']+"LVM-Med (R50)"+".pth" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('resnet50', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/lvmmed/lvmmed_resnet.torch',ours=None)
    main(hypar=hypar)
    
    # # #vitb-lvmmed
    hypar["restore_model"]=hypar['model_path']+"LVM-Med (Vitb)"+".pth" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('vit_base_patch16_224', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/lvmmed/lvmmed_vit.pth',ours=None)
    main(hypar=hypar)
    
    # #vitb-medsam
    hypar["restore_model"]=hypar['model_path']+"MedSAM (Vitb)"+".pth" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('vit_base_patch16_224', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/medsam_vit_b.pth',ours=None)
    main(hypar=hypar)
    
    #mammo-clip-b2
    hypar["restore_model"]=hypar['model_path']+"Mammo-CLIP (Enb2)"+".pth"
    hypar["model"]=MultiTaskModel('efficientnet', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/mammo-clip/b2-model-best-epoch-10.tar',ours=None)
    main(hypar=hypar)
    
    # # #mammo-clip-b5
    hypar["restore_model"]=hypar['model_path']+"Mammo-CLIP (Enb5)"+".pth" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('efficientnet', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/mammo-clip/b5-model-best-epoch-7.tar',ours=None)
    main(hypar=hypar)
    
    hypar["input_size"] = [518, 518]
    
    # # # #VersaMammo
    hypar["restore_model"]=hypar['model_path']+"VersaMammo"+".pth" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('vit_base_patch16_224', hypar['label_mappings'],checkpoint_path=None,ours='vitb14versamammo')
    main(hypar=hypar)
    
    # # #MaMA
    hypar['restore_model'] =hypar['model_path']+"MAMA (Vitb)"+".pth"
    hypar["model"]=MultiTaskModel('mama', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/mama_embed_pretrained_40k_steps_last.ckpt',ours=None)
    main(hypar=hypar)
    
