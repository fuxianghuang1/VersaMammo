import os
import time
import numpy as np
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from dataloader import myDataset,myNormalize
from torch.utils.data import DataLoader
from model import MultiTaskModel

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,roc_auc_score
from preprocess import preprocess
current_dir = os.path.dirname(os.path.abspath(__file__))

criterion = nn.BCELoss()
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
                labels = {task: label for task, label in labels.items()}
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

            # print("time lapse for data preparation: ", time.time()-start_read, ' s')

            # y zero the parameter gradients
            start_inf_loss_back = time.time()
            optimizer.zero_grad()
            ds= net(inputs_v)
            loss = 0
            for task, output in ds.items():
                loss += criterion(output, labels_v[task].squeeze(-1))

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            # running_tar_loss += loss2.item()

            # del outputs, loss
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
                print("last_acc:",last_acc)
                print("tmp_acc:",tmp_acc)
                # for fi in range(len(last_acc)):
                if(tmp_acc>last_acc):
                    tmp_out = 1
                print("tmp_out:",tmp_out)
                if(tmp_out):
                    notgood_cnt = 0
                    last_acc = tmp_acc
                    besacc = str(round(tmp_acc,4))
                    torch.save(net.state_dict(), os.path.join(model_path,hypar['model_name'] + '.pth'))

                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0

                if(notgood_cnt >= hypar["early_stop"]):
                    print("No improvements in the last "+str(notgood_cnt)+" validation periods, so training stopped !")
                    early_stop_triggered = True
                    break

        if early_stop_triggered:
            break
    print("Training Reaches The Maximum Epoch Number")

def valid(net, valid_dataloader, hypar, epoch=0):
    net.eval()
    print("Validating...")
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

    # Calculate average metrics across all tasks
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
        print("--- create training dataloader ---")
        train_dataset=myDataset(hypar['train_datapath'],[myNormalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])],hypar['label_mappings'])
        hypar['train_num']=len(train_dataset)
        train_dataloader=DataLoader(train_dataset, batch_size=hypar["batch_size_train"], shuffle=True, num_workers=8, pin_memory=False)
    print("--- create validation dataloader ---")

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

    print("--- define optimizer ---")
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
    hypar['dataset']='DMID-finding'
    hypar['task']='Finding'
    hypar['input_path']=f'../../datapre/classification_data/{hypar["dataset"]}'
    hypar['finetune']='lp'#lp or ft
    hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
    hypar["seed"] = 0
    hypar["start_ite"]=0

    torch.manual_seed(hypar["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hypar["seed"])

    hypar['gpu_id']=[0]
        
    hypar["model_path"]=f"{current_dir}/saved_model/{hypar['dataset']}/{hypar['task']}/{hypar['finetune']}/"
    
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
    print("batch size: ", hypar["batch_size_train"])

    hypar["max_ite"] = 10000000 ## if early stop couldn't stop the training process, stop it by the max_ite_num
    hypar["max_epoch_num"] = 1000000 ## if early stop and max_ite couldn't stop the training process, stop it by the max_epoch_num
    
    info={hypar['task']:data_info[hypar['dataset']][hypar['task']]}
    
    hypar["input_size"] = [512, 512] 
    hypar['train_datapath'] = hypar['input_path']+'/Train_cache_'+str(hypar["input_size"][0])
    if not os.path.exists(hypar['train_datapath']):
        preprocess(hypar['input_path']+'/Train',hypar['train_datapath'],hypar["input_size"])
    hypar['val_datapath'] = hypar['input_path']+'/Eval_cache_'+str(hypar["input_size"][0])
    if not os.path.exists(hypar['val_datapath']):
        preprocess(hypar['input_path']+'/Eval',hypar['val_datapath'],hypar["input_size"])
        
    # # # #resnet50-lvmmed
    hypar['model_name'] ="LVM-Med (R50)" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('resnet50', info,checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/LVM-Med (R50).torch',ours=None)
    main(hypar=hypar)
    
    # # # #vitb-lvmmed
    hypar['model_name'] ="LVM-Med (ViT-B)" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('vit_base_patch16_224', info,checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/LVM-Med (ViT-B).pth',ours=None)
    main(hypar=hypar)
    
    # # #vitb-medsam
    hypar['model_name'] ="MedSAM (ViT-B)" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('vit_base_patch16_224', info,checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/MedSAM (ViT-B).pth',ours=None)
    main(hypar=hypar)
    
    # #mammo-clip-b2
    hypar['model_name']="Mammo-CLIP (Enb2)"
    hypar["model"]=MultiTaskModel('efficientnet', info,checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/Mammo-CLIP (Enb2).tar',ours=None)
    main(hypar=hypar)
    
    # # # #mammo-clip-b5
    hypar['model_name'] ="Mammo-CLIP (Enb5)" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('efficientnet', info,checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/Mammo-CLIP (Enb5).tar',ours=None)
    main(hypar=hypar)
    
    # # # #VersaMammo
    hypar['model_name']="VersaMammo (Enb5)"
    hypar["model"]=MultiTaskModel('efficientnet', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/VersaMammo (Enb5).pth',ours=None)
    main(hypar=hypar)
    
    hypar["input_size"] = [512, 512] 
    hypar['train_datapath'] = hypar['input_path']+'/Train_cache_'+str(hypar["input_size"][0])
    if not os.path.exists(hypar['train_datapath']):
        preprocess(hypar['input_path']+'/Train',hypar['train_datapath'],hypar["input_size"])
    hypar['val_datapath'] = hypar['input_path']+'/Eval_cache_'+str(hypar["input_size"][0])
    if not os.path.exists(hypar['val_datapath']):
        preprocess(hypar['input_path']+'/Eval',hypar['val_datapath'],hypar["input_size"])
    
    # # #MAMA
    hypar['model_name']="MAMA (ViT-B)"
    hypar["model"]=MultiTaskModel('MAMA', hypar['label_mappings'],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/MAMA (ViT-B).ckpt',ours=None)
    main(hypar=hypar)
