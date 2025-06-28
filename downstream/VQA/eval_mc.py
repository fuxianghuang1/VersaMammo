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
from model_mc import MultiTaskModel
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,roc_auc_score
import glob
import re
import contextlib
from sklearn.utils import resample
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
current_dir = os.path.dirname(os.path.abspath(__file__))
def get_path(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.pth"))
    pattern = re.compile(r"gpu_itr_(\d+)_traLoss")

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
        
        probs = {task: logits[task].detach() for task in logits}
        preds = {task: (logits[task] > 0.5).float() for task in logits}
        for task, output in logits.items():
            all_probs[task].append(probs[task].cpu().numpy())
            all_preds[task].append(preds[task].cpu().numpy())
            all_labels[task].append(label.cpu().numpy())
            for i in range(label.size(0)):
                indices = torch.nonzero(label[i] == 1)[-1].cpu().tolist()
                for key in indices:
                    class_counts[task][key]+=1
                
                for key in indices:
                    if preds[task][i][key]==label[i][key]:
                        correct_counts[task][key] += 1

    for task in all_preds:
        all_probs[task] = np.vstack(all_probs[task])
        all_preds[task] = np.vstack(all_preds[task])
        all_labels[task] = np.vstack(all_labels[task])

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

        accuracy_bs = accuracy_score(y_true_bs, y_pred_bs)
        f1_bs = f1_score(y_true_bs, y_pred_bs, average='macro')

        non_zero_indices_bs = np.sum(y_true_bs, axis=0)!= 0
        if np.sum(non_zero_indices_bs) == 0:
            auc_bs = 0
        else:
            auc_bs = roc_auc_score(y_true_bs[:, non_zero_indices_bs], y_prob_bs[:, non_zero_indices_bs], average='macro')
        accuracy_scores.append(accuracy_bs)
        f1_scores.append(f1_bs)
        auc_scores.append(auc_bs)

    accuracy_ci = np.percentile(accuracy_scores, [2.5, 97.5])
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

    print('============================')
    print(f'Average metrics across all tasks - Acc: {avg_acc:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, AUC: {avg_auc:.4f}')
    print(f'Ci - Acc_ci: [{accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f}], F1_ci: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}], Auc_ci: [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]')
    print('\n')
    return avg_acc, i_val, tmp_time


def main(hypar): # model: "train", "test"
    val_dataset=MammoVQA_image(hypar['val_data'],hypar['label_mappings'],type='multiple')
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
    hypar['question topic']='Abnormality'
    hypar['finetune']='lp'#lp or ft
    hypar["valid_out_dir"] = f"{os.path.dirname(current_dir)}/VQA/results/{hypar['question topic']}/{hypar['finetune']}"+'.txt'##"../DIS5K-Results-test" ## output inferenced segmentation maps into this fold
    hypar['gpu_id']=[0]

    hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
    hypar["seed"] = 0
    hypar["start_ite"]=0

    torch.manual_seed(hypar["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hypar["seed"])

    hypar["input_size"] = [518, 518] 
    hypar['val_datapath'] = f'{current_dir}/MammoVQA-Image-Bench.json'
    with open(hypar['val_datapath'], 'r') as f:
        val_data = json.load(f)
    hypar['val_data'] = [{"ID": key, **value} for key, value in val_data.items() if value["Question topic"] == hypar['question topic']]
    hypar["model_path"]=f"{current_dir}/saved_model/{hypar['question topic']}/{hypar['finetune']}/"
    
    data_info = {
        'Abnormality': {'Architectural distortion':0,'Asymmetry':1,'Calcification':2,'Mass':3,'Miscellaneous':4,'Nipple retraction':5,'Normal':6,'Skin retraction':7,'Skin thickening':8,'Suspicious lymph node':9},
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

    hypar["batch_size_valid"] = 1

    hypar["max_ite"] = 10000000 ## if early stop couldn't stop the training process, stop it by the max_ite_num
    
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
    hypar['restore_model'] =hypar['model_path']+"MAMA"+".pth"
    hypar["model"]=MultiTaskModel('mama', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/mama_embed_pretrained_40k_steps_last.ckpt',ours=None)
    main(hypar=hypar)