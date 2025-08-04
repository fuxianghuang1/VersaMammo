import os
import time
import numpy as np
from skimage import io
import time

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from dataloader import myDataset,myNormalize,myRandomHFlip,myRandomVFlip
from torch.utils.data import DataLoader
from basics import f1_mae_torch, dice_torch #normPRED, GOSPRF1ScoresCache,f1score_torch,
from preprocess import preprocess
from TransUNet.vit_seg_modeling import VisionTransformer 
from UNetResNet50 import UNetResNet50
from UNetEfficientNetB2 import UNetEfficientNetB2
from UNetEfficientNetB5 import UNetEfficientNetB5
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
current_dir = os.path.dirname(os.path.abspath(__file__))

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()

        # Calculate Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)

        # Dice loss
        dice_loss = 1 - dice_coeff

        return dice_loss
dice_loss=DiceLoss()
bce_loss = nn.BCELoss(size_average=True)
def muti_loss_fusion(preds, target):
    loss0 = 0.0
    loss = 0.0
    loss =bce_loss(preds,target)+dice_loss(preds,target)
    loss0=loss
    return loss0, loss
def train(net, optimizer, train_dataloader, valid_dataloader, hypar): #model_path, model_save_fre, max_ite=1000000:

    model_path = hypar["model_path"]
    model_save_fre = hypar["model_save_fre"]
    max_ite = hypar["max_ite"]
    batch_size_train = hypar["batch_size_train"]
    batch_size_valid = hypar["batch_size_valid"]

    if(not os.path.exists(os.path.dirname(model_path))):
        os.makedirs(os.path.dirname(model_path))

    ite_num = hypar["start_ite"] # count the toal iteration number
    ite_num4val = 0 #
    running_loss = 0.0 # count the toal loss
    running_tar_loss = 0.0 # count the target output loss
    last_f1 = [0]
    last_dice=[0]

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
            imidx,image_name,inputs, labels = data['imidx'],data['image_name'],data['images'], data['masks']

            if(hypar["model_digit"]=="full"):
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
            else:
                inputs = inputs.type(torch.HalfTensor)
                labels = labels.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(hypar['gpu_id'][0]), requires_grad=False), Variable(labels.cuda(hypar['gpu_id'][0]), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # print("time lapse for data preparation: ", time.time()-start_read, ' s')

            # y zero the parameter gradients
            start_inf_loss_back = time.time()
            optimizer.zero_grad()
            ds= net(inputs_v)
            loss2, loss = muti_loss_fusion(ds, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del outputs, loss
            del ds, loss2, loss
            end_inf_loss_back = time.time()-start_inf_loss_back

            print(">>>"+model_path.split('/')[-1]+" - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, time-per-iter: %3f s, time_read: %3f" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val, time.time()-start_last, time.time()-start_last-end_inf_loss_back))
            start_last = time.time()

            if ite_num % model_save_fre == 0:  # validate every 2000 iterations
                notgood_cnt += 1
                net.eval()
                tmp_dice, val_loss, tar_loss, i_val, tmp_time = valid(net, valid_dataloader,hypar, epoch)
                net.train()  # resume train

                tmp_out = 0
                print("last_dice:",last_dice)
                print("tmp_dice:",tmp_dice)
                for fi in range(len(last_dice)):
                    if(tmp_dice[fi]>last_dice[fi]):
                        tmp_out = 1
                print("tmp_out:",tmp_out)
                if(tmp_out):
                    notgood_cnt = 0
                    last_dice = tmp_dice
                    tmp_dice_str = [str(round(f1x,4)) for f1x in tmp_dice]
                    # tmp_mae_str = [str(round(mx,4)) for mx in tmp_mae]
                    maxdice = '_'.join(tmp_dice_str)
                    torch.save(net.state_dict(), model_path + '.pth')

                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0

                if(notgood_cnt >= hypar["early_stop"]):
                    print("No improvements in the last "+str(notgood_cnt)+" validation periods, so training stopped !")
                    early_stop_triggered = True
                    break

        if early_stop_triggered:
            break
        # scheduler.step()
    print("Training Reaches The Maximum Epoch Number")

def valid(net, valid_dataloader, hypar, epoch=0):
    net.eval()
    print("Validating...")
    epoch_num = hypar["max_epoch_num"]

    val_loss = 0.0
    tar_loss = 0.0
    val_cnt = 0.0

    tmp_f1 = []
    tmp_mae = []
    tmp_time = []
    tmp_dice = []

    start_valid = time.time()

    val_num = hypar['val_num']
    mybins = np.arange(0,256)
    PRE = np.zeros((val_num,len(mybins)-1))
    REC = np.zeros((val_num,len(mybins)-1))
    F1 = np.zeros((val_num,len(mybins)-1))
    MAE = np.zeros((val_num))
    DICE = np.zeros((val_num))
    
    for i_val, data_val in enumerate(valid_dataloader):
        val_cnt = val_cnt + 1.0
        imidx,image_name,inputs_val, labels_val = data_val['imidx'],data_val['image_name'],data_val['images'], data_val['masks']

        if(hypar["model_digit"]=="full"):
            inputs_val = inputs_val.type(torch.FloatTensor)
            labels_val = labels_val.type(torch.FloatTensor)
        else:
            inputs_val = inputs_val.type(torch.HalfTensor)
            labels_val = labels_val.type(torch.HalfTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_val_v, labels_val_v = Variable(inputs_val.cuda(hypar['gpu_id'][0]), requires_grad=False), Variable(labels_val.cuda(hypar['gpu_id'][0]), requires_grad=False)
        else:
            inputs_val_v, labels_val_v = Variable(inputs_val, requires_grad=False), Variable(labels_val,requires_grad=False)

        t_start = time.time()
        ds_val = net(inputs_val_v)
        t_end = time.time()-t_start
        tmp_time.append(t_end)

        loss2_val, loss_val = muti_loss_fusion(ds_val, labels_val_v)

        gt = np.squeeze(io.imread(os.path.join('/'.join(hypar['val_datapath'].split('/')[:-1]),hypar['val_datapath'].split('/')[-1].split('_')[0],image_name[0],'mask.png'))) # max = 255
        if gt.max()==1:
            gt=gt*255
        with torch.no_grad():
            gt = torch.tensor(gt).cuda(hypar['gpu_id'][0])

        pred_val = ds_val[0,:,:,:] # B x 1 x H x W
        pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),gt.shape,mode='bilinear'))

        dice = dice_torch(pred_val*255, gt)

        if hypar["plot_output"]:
            if(hypar["valid_out_dir"]!=""):
                if(not os.path.exists(hypar["valid_out_dir"])):
                    os.makedirs(hypar["valid_out_dir"])
                io.imsave(os.path.join(hypar["valid_out_dir"],image_name[0]+".png"),(pred_val*255).cpu().data.numpy().astype(np.uint8))
            print(image_name[0]+".png")

        DICE[imidx[0]] = dice

        del ds_val, gt
        gc.collect()
        torch.cuda.empty_cache()

        val_loss += loss_val.item()
        tar_loss += loss2_val.item()

        print("[validating: %5d/%5d] val_ls:%f, tar_ls: %f, dice: %f, time: %f"% (i_val, val_num, val_loss / (i_val + 1), tar_loss / (i_val + 1),  DICE[imidx[0]],t_end))

        del loss2_val, loss_val

    print('============================')
    tmp_dice.append(np.mean(DICE))
    print('Mean Dice:',np.mean(DICE))
    
    # print('Count:',count)

    return tmp_dice, val_loss, tar_loss, i_val, tmp_time

def main(hypar): # model: "train", "test"

    if(hypar["mode"]=="train"):
        print("--- create training dataloader ---")
        train_dataset=myDataset(hypar['train_datapath'],[myRandomVFlip(),myRandomHFlip(),myNormalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

        hypar['train_num']=len(train_dataset)
        train_dataloader=DataLoader(train_dataset, batch_size=hypar["batch_size_train"], shuffle=True, num_workers=8, pin_memory=False)
    print("--- create validation dataloader ---")
    val_dataset=myDataset(hypar['val_datapath'],[myNormalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
    hypar['val_num']=len(val_dataset)
    val_dataloader=DataLoader(val_dataset, batch_size=hypar["batch_size_valid"], shuffle=False, num_workers=1, pin_memory=False)

    print("--- build model ---")
    net = hypar["model"]

    # convert to half precision
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
    hypar['dataset']='CBIS-DDSM'
    hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
    hypar["seed"] = 0

    torch.manual_seed(hypar["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hypar["seed"])

    hypar['gpu_id']=[0]

    print("building model...")
    hypar["early_stop"] = 20 ## stop the training when no improvement in the past 20 validation periods, smaller numbers can be used here e.g., 5 or 10.
    hypar["model_save_fre"] = 500 ## valid and save model weights every 2000 iterations

    hypar["batch_size_train"] = 8 ## batch size for training
    hypar["batch_size_valid"] = 1 ## batch size for validation and inferencing
    print("batch size: ", hypar["batch_size_train"])

    hypar["max_ite"] = 10000000 ## if early stop couldn't stop the training process, stop it by the max_ite_num
    hypar["max_epoch_num"] = 1000000 ## if early stop and max_ite couldn't stop the training process, stop it by the max_epoch_num
    
    hypar["input_size"] = [512, 512] 
    input_path = f'../../datapre/segdetdata/{hypar["dataset"]}/Train'  # 替换为你的实际路径
    hypar['train_datapath'] = input_path+'_cache_'+str(hypar["input_size"][0])
    if not os.path.exists(hypar['train_datapath']):
        preprocess(input_path,hypar['train_datapath'],hypar["input_size"])
    input_path = f'../../datapre/segdetdata/{hypar["dataset"]}/Eval'  # 替换为你的实际路径
    hypar['val_datapath'] = input_path+'_cache_'+str(hypar["input_size"][0])
    if not os.path.exists(hypar['val_datapath']):
        preprocess(input_path,hypar['val_datapath'],hypar["input_size"])
    
    
    # # # #efficientnetb2-Mammo-Clip
    if hypar["mode"] == "train":
        hypar["valid_out_dir"] = "" ## for "train" model leave it as "", for "valid"("inference") mode: set it according to your local directory
        hypar["model_path"] =f"{os.path.dirname(current_dir)}/Segment/saved_model/{hypar['dataset']}/Mammo-CLIP (Enb2)" ## model weights saving (or restoring) path
        hypar["restore_model"] = "" ## name of the segmentation model weights .pth for resume training process from last stop or for the inferencing
        hypar["start_ite"] = 0 ## start iteration for the training, can be changed to match the restored training process
        hypar["plot_output"]=False
    else:
        hypar["valid_out_dir"] = f"{os.path.dirname(current_dir)}/Segment/results/{hypar['dataset']}/Mammo-CLIP (Enb2)"##"../DIS5K-Results-test" ## output inferenced segmentation maps into this fold
        hypar["restore_model"] = ""
    hypar["model"]=UNetEfficientNetB2(checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/mammo-clip/b2-model-best-epoch-10.tar',pretrained=True)
    main(hypar=hypar)
    
    # #     # # # #efficientnetb5-Mammo-Clip
    if hypar["mode"] == "train":
        hypar["valid_out_dir"] = "" ## for "train" model leave it as "", for "valid"("inference") mode: set it according to your local directory
        hypar["model_path"] =f"{os.path.dirname(current_dir)}/Segment/saved_model/{hypar['dataset']}/Mammo-CLIP (Enb5)" ## model weights saving (or restoring) path
        hypar["restore_model"] = "" ## name of the segmentation model weights .pth for resume training process from last stop or for the inferencing
        hypar["start_ite"] = 0 ## start iteration for the training, can be changed to match the restored training process
        hypar["plot_output"]=False
    else:
        hypar["valid_out_dir"] = f"{os.path.dirname(current_dir)}/Segment/results/{hypar['dataset']}/Mammo-CLIP (Enb5)"##"../DIS5K-Results-test" ## output inferenced segmentation maps into this fold
        hypar["restore_model"] = ""
    hypar["model"]=UNetEfficientNetB5(checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/mammo-clip/b5-model-best-epoch-7.tar',pretrained=True)
    main(hypar=hypar)
    
        # # # #VersaMammo
    if hypar["mode"] == "train":
        hypar["valid_out_dir"] = "" ## for "train" model leave it as "", for "valid"("inference") mode: set it according to your local directory
        hypar["model_path"] =f"{os.path.dirname(current_dir)}/Segment/saved_model/{hypar['dataset']}/VersaMammo" ## model weights saving (or restoring) path
        hypar["restore_model"] = "" ## name of the segmentation model weights .pth for resume training process from last stop or for the inferencing
        hypar["start_ite"] = 0 ## start iteration for the training, can be changed to match the restored training process
        hypar["plot_output"]=False
    else:
        hypar["valid_out_dir"] = f"{os.path.dirname(current_dir)}/Segment/results/{hypar['dataset']}/VersaMammo"##"../DIS5K-Results-test" ## output inferenced segmentation maps into this fold
        hypar["restore_model"] = ""
    hypar["model"]=UNetEfficientNetB5(checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/Ours/ENb5/ENB5_SL.pth',pretrained=True)
    main(hypar=hypar)
    
    
    # # # #resnet50-lvmmed
    if hypar["mode"] == "train":
        hypar["valid_out_dir"] = "" ## for "train" model leave it as "", for "valid"("inference") mode: set it according to your local directory
        hypar["model_path"] =f"{os.path.dirname(current_dir)}/Segment/saved_model/{hypar['dataset']}/LVM-Med (R50)" ## model weights saving (or restoring) path
        hypar["restore_model"] = "" ## name of the segmentation model weights .pth for resume training process from last stop or for the inferencing
        hypar["start_ite"] = 0 ## start iteration for the training, can be changed to match the restored training process
        hypar["plot_output"]=False
    else:
        hypar["valid_out_dir"] = f"{os.path.dirname(current_dir)}/Segment/results/{hypar['dataset']}/LVM-Med (R50)"##"../DIS5K-Results-test" ## output inferenced segmentation maps into this fold
        hypar["restore_model"] = ""
    hypar["model"]=UNetResNet50(checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/lvmmed/lvmmed_resnet.torch',pretrained=True)
    main(hypar=hypar)
    
    # # #vitb-lvmmed
    if hypar["mode"] == "train":
        hypar["valid_out_dir"] = "" ## for "train" model leave it as "", for "valid"("inference") mode: set it according to your local directory
        hypar["model_path"] =f"{os.path.dirname(current_dir)}/Segment/saved_model/{hypar['dataset']}/LVM-Med (Vitb)" ## model weights saving (or restoring) path
        hypar["restore_model"] = "" ## name of the segmentation model weights .pth for resume training process from last stop or for the inferencing
        hypar["start_ite"] = 0 ## start iteration for the training, can be changed to match the restored training process
        hypar["plot_output"]=False
    else:
        hypar["valid_out_dir"] = f"{os.path.dirname(current_dir)}/Segment/results/{hypar['dataset']}/LVM-Med (Vitb)"##"../DIS5K-Results-test" ## output inferenced segmentation maps into this fold
        hypar["restore_model"] = ""
    hypar['checkpoint_path']=f'{os.path.dirname(current_dir)}/Sotas/lvmmed/lvmmed_vit.pth'
    hypar["model"]=VisionTransformer(img_size=hypar["input_size"],checkpoint_path=hypar['checkpoint_path'])
    main(hypar=hypar)
    
    # # #vitb-medsam
    if hypar["mode"] == "train":
        hypar["valid_out_dir"] = "" ## for "train" model leave it as "", for "valid"("inference") mode: set it according to your local directory
        hypar["model_path"] =f"{os.path.dirname(current_dir)}/Segment/saved_model/{hypar['dataset']}/MedSAM (Vitb)" ## model weights saving (or restoring) path
        hypar["restore_model"] = "" ## name of the segmentation model weights .pth for resume training process from last stop or for the inferencing
        hypar["start_ite"] = 0 ## start iteration for the training, can be changed to match the restored training process
        hypar["plot_output"]=False
    else:
        hypar["valid_out_dir"] = f"{os.path.dirname(current_dir)}/Segment/results/{hypar['dataset']}/MedSAM (Vitb)"##"../DIS5K-Results-test" ## output inferenced segmentation maps into this fold
        hypar["restore_model"] = ""
    hypar['checkpoint_path']=f'{os.path.dirname(current_dir)}/Sotas/medsam_vit_b.pth'
    hypar["model"]=VisionTransformer(img_size=hypar["input_size"],checkpoint_path=hypar['checkpoint_path'])
    main(hypar=hypar)

    # # # # #MAMA
    if hypar["mode"] == "train":
        hypar["valid_out_dir"] = "" ## for "train" model leave it as "", for "valid"("inference") mode: set it according to your local directory
        hypar["model_path"] =f"{os.path.dirname(current_dir)}/Segment/saved_model/{hypar['dataset']}/MAMA (Vitb)" ## model weights saving (or restoring) path
        hypar["restore_model"] = "" ## name of the segmentation model weights .pth for resume training process from last stop or for the inferencing
        hypar["start_ite"] = 0 ## start iteration for the training, can be changed to match the restored training process
        hypar["plot_output"]=False
    else:
        hypar["valid_out_dir"] = f"{os.path.dirname(current_dir)}/Segment/results/{hypar['dataset']}/MAMA (Vitb)"##"../DIS5K-Results-test" ## output inferenced segmentation maps into this fold
        hypar["restore_model"] = ""
    hypar["model"]=VisionTransformer(img_size=hypar["input_size"],checkpoint_path=f'{os.path.dirname(current_dir)}/Sotas/mama_embed_pretrained_40k_steps_last.ckpt',ours=None)
    main(hypar=hypar)
