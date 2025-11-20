import os
import time
import numpy as np
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from dataloader import myDataset,myNormalize
from torch.utils.data import DataLoader
from model import MultiTaskModel

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from preprocess import preprocess
from sklearn.preprocessing import OneHotEncoder
current_dir = os.path.dirname(os.path.abspath(__file__))


criterion = nn.CrossEntropyLoss()
def train(net, optimizer, train_dataloader, valid_dataloader, hypar): #model_path, model_save_fre, max_ite=1000000:
    model_path = hypar["model_path"]
    model_save_fre = hypar["model_save_fre"]
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
    for epoch in range(epoch_num): ## set the epoch num as 100000

        for i, data in enumerate(gos_dataloader):

            if(ite_num >= max_ite):
                print("Training Reached the Maximal Iteration Number ", max_ite)
                exit()

            # start_read = time.time()
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            # get the inputs
            imidx,image_name,inputs, labels = data['imidx'],data['image_name'],data['images'], data['labels']

            if(hypar["model_digit"]=="full"):
                inputs = inputs.type(torch.FloatTensor)
                labels = {task: label.type(torch.long) for task, label in labels.items()}
            # else:
            #     inputs = inputs.type(torch.HalfTensor)
            #     labels = {task: label.type(torch.halflong) for task, label in labels.items()}

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v = Variable(inputs.cuda(hypar['gpu_id'][0]), requires_grad=False)
                labels_v = {task: Variable(label.cuda(hypar['gpu_id'][0]), requires_grad=False) for task, label in labels.items()}
            else:
                inputs_v = Variable(inputs, requires_grad=False)
                labels_v = {task: Variable(label, requires_grad=False) for task, label in labels.items()}

            start_inf_loss_back = time.time()
            optimizer.zero_grad()
            ds= net(inputs_v)
            loss = 0
            for task, output in ds.items():
                loss += criterion(output, labels_v[task].squeeze(-1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            del loss
            end_inf_loss_back = time.time()-start_inf_loss_back

            print(">>>"+hypar['model_name']+" - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f,  time-per-iter: %3f s, time_read: %3f" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,  time.time()-start_last, time.time()-start_last-end_inf_loss_back))
            start_last = time.time()

            if ite_num % model_save_fre == 0:  # validate every 2000 iterations
                notgood_cnt += 1
                net.eval()
                tmp_acc,  i_val, tmp_time = valid(net, valid_dataloader,hypar, epoch)
                net.train()  # resume train

                tmp_out = 0
                if(tmp_acc>last_acc):
                    tmp_out = 1
                if(tmp_out):
                    notgood_cnt = 0
                    last_acc = tmp_acc
                    besacc = str(round(tmp_acc,4))
                    torch.save(net.state_dict(), os.path.join(model_path,hypar['model_name'] + '.pth'))

                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0

                if(notgood_cnt >= hypar["early_stop"]):
                    # print("No improvements in the last "+str(notgood_cnt)+" validation periods, so training stopped !")
                    early_stop_triggered = True
                    break

        if early_stop_triggered:
            break

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

            batch_acc = accuracy_score(labels_val_v[task].cpu().numpy(), preds.cpu().numpy())
            batch_precision = precision_score(labels_val_v[task].cpu().numpy(), preds.cpu().numpy(), average='micro', zero_division=0)
            batch_recall = recall_score(labels_val_v[task].cpu().numpy(), preds.cpu().numpy(), average='micro')
            batch_f1 = f1_score(labels_val_v[task].cpu().numpy(), preds.cpu().numpy(), average='micro')

            batch_metrics.append((batch_acc, batch_precision, batch_recall, batch_f1))
            print(f'{i_val}/{val_num} for {task} - Acc: {batch_acc:.4f}, Precision: {batch_precision:.4f}, Recall: {batch_recall:.4f}, F1: {batch_f1:.4f}')

        gc.collect()
        torch.cuda.empty_cache()

    total_acc, total_precision, total_recall, total_f1, total_auc = 0, 0, 0, 0, 0
    num_tasks = len(net.classifiers.keys())
    for task in net.classifiers.keys():
        all_preds[task] = np.array(all_preds[task])
        all_labels[task] = np.array(all_labels[task])
        all_probs[task] = np.array(all_probs[task])

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
        
        total_acc += epoch_acc
        total_precision += epoch_precision
        total_recall += epoch_recall
        total_f1 += epoch_f1
        total_auc += epoch_auc

        print('============================')
        print(f'{task} - Acc: {epoch_acc:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}, AUC: {epoch_auc:.4f}')
        print(f'Statistics for task {task}:')
        for label, count in class_counts[task].items():
            correct = correct_counts[task][label]
            print(f'  Label {hypar["reverse_label_mappings"][task][label]}: Total: {count}, Correct: {correct}')

    avg_acc = total_acc / num_tasks
    avg_precision = total_precision / num_tasks
    avg_recall = total_recall / num_tasks
    avg_f1 = total_f1 / num_tasks
    avg_auc = total_auc / num_tasks
    print(avg_auc)
    print('============================')
    print(f'Average metrics across all tasks - Acc: {avg_acc:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, AUC: {avg_auc:.4f}')

    return avg_acc, i_val, tmp_time

def main(hypar): 
    if(hypar["mode"]=="train"):
        train_dataset=myDataset(hypar['train_datapath'],[myNormalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])],hypar['label_mappings'])
        hypar['train_num']=len(train_dataset)
        train_dataloader=DataLoader(train_dataset, batch_size=hypar["batch_size_train"], shuffle=True, num_workers=8, pin_memory=False)
    val_dataset=myDataset(hypar['val_datapath'],[myNormalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])],hypar['label_mappings'])
    hypar['val_num']=len(val_dataset)
    val_dataloader=DataLoader(val_dataset, batch_size=hypar["batch_size_valid"], shuffle=False, num_workers=1, pin_memory=False)

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
            #   scheduler,
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
    hypar['dataset']='INbreast'
    hypar['task']='Bi-Rads'
    hypar['finetune']='lp'#lp or ft
    hypar['input_path']=f'{current_dir}/../data/INbreast-classification'
    print(hypar['input_path'])
    hypar['gpu_id']=[0]

    hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
    hypar["seed"] = 0
    hypar["start_ite"]=0

    torch.manual_seed(hypar["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hypar["seed"])
        
    hypar["model_path"]=f"{current_dir}/saved_model/{hypar['dataset']}/{hypar['task']}/{hypar['finetune']}/"

    data_info = {
        'INbreast': {
            'Composition': {'Level A': 0, 'Level B': 1, 'Level C': 2, 'Level D': 3},
            'Bi-Rads': {'Bi-Rads 1': 0, 'Bi-Rads 2': 1, 'Bi-Rads 3': 2, 'Bi-Rads 4': 3,  'Bi-Rads 5': 4}
        }
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
    """['resnet50', 'efficientnet_b2', 'vit_base_patch16_224']"""
    """['vitb14imagenet','vitb14dinov2mammo','vitb14dinov2mammo1']"""
    hypar["early_stop"] = 20 
    hypar["model_save_fre"] = 500 

    hypar["batch_size_train"] = 16 ## batch size for training
    hypar["batch_size_valid"] = 1 ## batch size for validation and inferencing
    hypar["max_ite"] = 10000000 ## if early stop couldn't stop the training process, stop it by the max_ite_num
    hypar["max_epoch_num"] = 1000000 ## if early stop and max_ite couldn't stop the training process, stop it by the max_epoch_num
    
    hypar["input_size"] = [512, 512] 
    hypar['train_datapath'] = hypar['input_path']+'/Train_cache_'+str(hypar["input_size"][0])
    if not os.path.exists(hypar['train_datapath']):
        preprocess(hypar['input_path']+'/Train',hypar['train_datapath'],hypar["input_size"])
    hypar['val_datapath'] = hypar['input_path']+'/Eval_cache_'+str(hypar["input_size"][0])
    if not os.path.exists(hypar['val_datapath']):
        preprocess(hypar['input_path']+'/Eval',hypar['val_datapath'],hypar["input_size"])
    
    # # #EfficientNet-ours
    hypar['model_name']="VersaMammo"
    hypar["model"]=MultiTaskModel('efficientnet', hypar['label_mappings'],checkpoint_path=f'{current_dir}/../../Sotas/VersaMammo/ENb5/ENB5_SL.pth',ours=None,finetune=hypar['finetune'])
    main(hypar=hypar)
