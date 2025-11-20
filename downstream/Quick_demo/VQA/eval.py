import os
import time
import numpy as np
from skimage import io
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
    
    for i_val, data_val in enumerate(valid_dataloader):
        val_cnt = val_cnt + 1.0
        image, question, label, image_path = data_val

        t_start = time.time()
        logits, loss= net(image, question, label,hypar["input_size"])
        label=torch.tensor(label)
        batch_metrics = []
        for task, output in logits.items():
            preds = torch.argmax(output, dim=1).item()
            output=[hypar['reverse_label_mappings'][task][preds]]
            imagepath='/'.join(image_path[0].split('/')[-3:])
            print(f'Image path: {imagepath}, Question: {question[0]}, Answer: {output[0]}')


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
  
    valid(net, val_dataloader, hypar)


if __name__ == "__main__":
    hypar = {}
    hypar["mode"] = "eval"
    hypar['question topic']='ACR'
    hypar['finetune']='lp'#lp or ft
    hypar["valid_out_dir"] = f"{os.path.dirname(current_dir)}/VQA/results/{hypar['question topic']}/{hypar['finetune']}"+'.txt'##"../DIS5K-Results-test" ## output inferenced segmentation maps into this fold
    hypar['gpu_id']=[0]
    hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
    hypar["seed"] = 0
    hypar["start_ite"]=0

    torch.manual_seed(hypar["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hypar["seed"])

    hypar['val_datapath'] = f'{current_dir}/../data/INbreast-VQA.json'
    with open(hypar['val_datapath'], 'r') as f:
        val_data = json.load(f)
    hypar['val_data'] = [{"ID": key, **value} for key, value in val_data.items() if value["Question topic"] == hypar['question topic']]
    hypar["model_path"]=f"{current_dir}/"
    print(hypar["model_path"])

    data_info = {
        'ACR': {'Level A':0,'Level B':1,'Level C':2,'Level D':3},
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
    
    hypar["input_size"] = [518, 518]
    
    
    # # # #VersaMammo
    hypar["restore_model"]=hypar['model_path']+"VersaMammo_VQA"+".pth" ## model weights saving (or restoring) path
    hypar["model"]=MultiTaskModel('vit_base_patch16_224', hypar['label_mappings'],checkpoint_path=None,ours='vitb14rand')
    main(hypar=hypar)

