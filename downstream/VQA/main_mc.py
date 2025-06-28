import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import numpy as np
from skimage import io
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from Mammo_VQA_dataset import MammoVQA_image,custom_collate_fn
from torch.utils.data import DataLoader
from model_mc import MultiTaskModel
import json

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
current_dir = os.path.dirname(os.path.abspath(__file__))

criterion = nn.CrossEntropyLoss()
def train(net, optimizer, train_dataloader, valid_dataloader, hypar): #model_path, model_save_fre, max_ite=1000000:
    model_path = hypar["model_path"]
    # model_save_fre = hypar["model_save_fre"]
    max_ite = hypar["max_ite"]
    batch_size_train = hypar["batch_size_train"]
    batch_size_valid = hypar["batch_size_valid"]

    if(not os.path.exists(model_path)):
        os.makedirs(model_path)

    ite_num = hypar["start_ite"] # count the toal iteration number
    ite_num4val = 0 #
    running_loss = 0.0 # count the toal loss
    running_tar_loss = 0.0 # count the target output loss
    last_f1 = [0]
    last_acc=0

    train_num = hypar['train_num']

    net.train()

    start_last = time.time()
    gos_dataloader = train_dataloader
    epoch_num = hypar["max_epoch_num"]
    notgood_cnt = 0
    early_stop_triggered = False
    accumulated_steps = 0  

    for epoch in range(epoch_num): 
        for i, data in enumerate(gos_dataloader):

            if ite_num >= max_ite:
                print("Training Reached the Maximal Iteration Number ", max_ite)
                exit()

            ite_num += 1
            ite_num4val += 1
            image, question, label, idx = data

            if accumulated_steps == 0:
                optimizer.zero_grad()

            start_inf_loss_back = time.time()
            logits, loss = net(image, question, label, hypar["input_size"])

            loss = loss / hypar["grad_accumulate"]
            loss.backward()
            accumulated_steps += 1

            if accumulated_steps == hypar["grad_accumulate"]:
                optimizer.step()
                accumulated_steps = 0

            running_loss += loss.item() * hypar["grad_accumulate"] 
            del loss
            end_inf_loss_back = time.time() - start_inf_loss_back

            print(">>>"+hypar['model_name']+" - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f,  time-per-iter: %3f s, time_read: %3f" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,  time.time()-start_last, time.time()-start_last-end_inf_loss_back))
            start_last = time.time()

        net.eval()
        tmp_acc,  i_val, tmp_time = valid(net, valid_dataloader,hypar, epoch)
        net.train()  # resume train

        tmp_out = 0

        if(tmp_acc>last_acc):
            tmp_out = 1
        if(tmp_out):

            last_acc = tmp_acc
            torch.save(net.state_dict(), os.path.join(model_path,hypar['model_name'] + '.pth'))
        else:
            break

        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0

def valid(net, valid_dataloader, hypar, epoch=0):
    net.eval()
    # print("Validating...")
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

        print('============================')
        print(f'{task} - Acc: {epoch_acc:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}, AUC: {epoch_auc:.4f}')

        print(f'Statistics for task {task}:')
        reverse_label_mapping = {v: k for k, v in hypar['label_mappings'][task].items()}
        for label, count in class_counts[task].items():
            correct = correct_counts[task][label]
            label_name = reverse_label_mapping[label]
            print(f'  Label {label_name}: Total: {count}, Correct: {correct}')
    avg_acc = total_acc / num_tasks
    avg_precision = total_precision / num_tasks
    avg_recall = total_recall / num_tasks
    avg_f1 = total_f1 / num_tasks
    avg_auc = total_auc / num_tasks

    print('============================')
    print(f'Average metrics across all tasks - Acc: {avg_acc:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, AUC: {avg_auc:.4f}')
    return avg_acc, i_val, tmp_time

def main(hypar): # model: "train", "test"

    if(hypar["mode"]=="train"):
     
        train_dataset=MammoVQA_image(hypar['train_data'],hypar['label_mappings'],type='multiple')
        hypar['train_num']=len(train_dataset)
        train_dataloader=DataLoader(train_dataset, collate_fn=custom_collate_fn, batch_size=hypar["batch_size_train"], shuffle=True, num_workers=8, pin_memory=False)
   
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
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if(hypar["mode"]=="train"):
        train(net,
              optimizer,
              train_dataloader,
              val_dataloader,
              hypar)
    else:
        valid(net,
              val_dataloader,
              hypar)


if __name__ == "__main__":
    hypar = {}
    hypar["mode"] = "train"
    hypar['question topic']='Abnormality'
    hypar['finetune']='lp'#lp or ft
    hypar['gpu_id']=[0]
    hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
    hypar["seed"] = 0
    hypar["start_ite"]=0

    torch.manual_seed(hypar["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hypar["seed"])

    hypar['train_datapath'] = f'{current_dir}/MammoVQA-Image-Train.json'
    with open(hypar['train_datapath'], 'r') as f:
        train_data = json.load(f)
    hypar['train_data'] = [{"ID": key, **value} for key, value in train_data.items() if value["Question topic"] == hypar['question topic']]
    hypar['val_datapath'] = f'{current_dir}/MammoVQA-Image-Eval.json'
    with open(hypar['val_datapath'], 'r') as f:
        val_data = json.load(f)
    hypar['val_data'] = [{"ID": key, **value} for key, value in val_data.items() if value["Question topic"] == hypar['question topic']]
        
    hypar["model_path"]=f"{current_dir}/saved_model/{hypar['question topic']}/{hypar['finetune']}"

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

    hypar["batch_size_train"] = 1 ## batch size for training
    hypar["grad_accumulate"]=16
    hypar["batch_size_valid"] = 1 ## batch size for validation and inferencing
    # print("batch size: ", hypar["batch_size_train"])

    hypar["max_ite"] = 10000000 ## if early stop couldn't stop the training process, stop it by the max_ite_num
    hypar["max_epoch_num"] = 100 ## if early stop and max_ite couldn't stop the training process, stop it by the max_epoch_num
    
    hypar["input_size"] = [512, 512] 
    
    # # # # #resnet50-lvmmed
    hypar['model_name'] ="LVM-Med (R50)" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('resnet50', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/lvmmed/lvmmed_resnet.torch',ours=None,finetune=hypar['finetune'])
    main(hypar=hypar)
    
    #vitb-lvmmed
    hypar['model_name'] ="LVM-Med (Vitb)" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('vit_base_patch16_224', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/lvmmed/lvmmed_vit.pth',ours=None,finetune=hypar['finetune'])
    main(hypar=hypar)
    
    # # #vitb-medsam
    hypar['model_name'] ="MedSAM (Vitb)" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('vit_base_patch16_224', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/medsam_vit_b.pth',ours=None,finetune=hypar['finetune'])
    main(hypar=hypar)
    
    # # # #mammo-clip-b2
    hypar['model_name']="Mammo-CLIP (Enb2)"
    hypar["model"]=MultiTaskModel('efficientnet', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/mammo-clip/b2-model-best-epoch-10.tar',ours=None,finetune=hypar['finetune'])
    main(hypar=hypar)
    
    # # #mammo-clip-b5
    hypar['model_name'] ="Mammo-CLIP (Enb5)" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('efficientnet', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/mammo-clip/b5-model-best-epoch-7.tar',ours=None,finetune=hypar['finetune'])
    main(hypar=hypar)
    
    hypar["input_size"] = [518, 518] 
    
   # # # #VersaMammo
    hypar['model_name'] ="VersaMammo" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('vit_base_patch16_224', hypar['label_mappings'],checkpoint_path=None,ours='vitb14versamammo',finetune=hypar['finetune'])
    main(hypar=hypar)
    
    # # #MaMA
    hypar['model_name']="MAMA"
    hypar["model"]=MultiTaskModel('mama', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/mama_embed_pretrained_40k_steps_last.ckpt',ours=None,finetune=hypar['finetune'])
    main(hypar=hypar)