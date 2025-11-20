import os
import time
import numpy as np
from skimage import io
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch, gc
import torch.nn as nn
import torch.optim as optim
import torchvision
from dataloader import myDataset,myNormalize,myRandomVFlip,myRandomHFlip
from torch.utils.data import DataLoader
from Faster_R_CNN_FPN import get_model
from preprocess import preprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches

current_dir = os.path.dirname(os.path.abspath(__file__))

def rescale_boxes(boxes, original_size, target_size=(224, 224)):
    """
    Rescale bounding boxes from target_size to original_size.
    
    Args:
        boxes (Tensor[N, 4]): Bounding boxes in target size.
        original_size (tuple): Original image size (width, height).
        target_size (tuple): Target image size (width, height), default is (224, 224).
        
    Returns:
        Tensor[N, 4]: Rescaled bounding boxes in original size.
    """
    original_width, original_height = original_size
    target_width, target_height = target_size
    
    scale_x = original_width / target_width
    scale_y = original_height / target_height
    
    rescaled_boxes = boxes.clone()
    rescaled_boxes[:, 0] = boxes[:, 0] * scale_x  # xmin
    rescaled_boxes[:, 1] = boxes[:, 1] * scale_y  # ymin
    rescaled_boxes[:, 2] = boxes[:, 2] * scale_x  # xmax
    rescaled_boxes[:, 3] = boxes[:, 3] * scale_y  # ymax
    
    return rescaled_boxes
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou
def apply_nms(orig_prediction, score_thresh=0.5, iou_thresh=0.5):
    """Apply non-maximum suppression to avoid overlapping boxes and filter low score boxes."""
    # Filter out boxes with scores lower than the threshold
    high_score_idxs = orig_prediction['scores'] >= score_thresh
    filtered_boxes = orig_prediction['boxes'][high_score_idxs]
    filtered_scores = orig_prediction['scores'][high_score_idxs]
    filtered_labels = orig_prediction['labels'][high_score_idxs]

    # Apply NMS
    keep = torchvision.ops.nms(filtered_boxes, filtered_scores, iou_thresh)

    final_prediction = {}
    final_prediction['boxes'] = filtered_boxes[keep]
    final_prediction['scores'] = filtered_scores[keep]
    final_prediction['labels'] = filtered_labels[keep]

    return final_prediction
def plot_image_with_boxes(image_name, image, gt_boxes, pred_boxes, pred_labels=None, pred_scores=None):
    original_height, original_width = image.shape[:2]
    original_size = (original_width, original_height)

    # Rescale boxes to original image size
    gt_boxes = rescale_boxes(gt_boxes, original_size)
    pred_boxes = rescale_boxes(pred_boxes, original_size)

    if not os.path.exists(hypar['valid_out_dir']):
        os.makedirs(hypar['valid_out_dir'])

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image, cmap='gray')

    gt_color = 'red'
    for box in gt_boxes:
        box = box.cpu().detach().numpy()
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=gt_color, facecolor='none')
        ax.add_patch(rect)
    
    pred_color = 'blue'
    for i, box in enumerate(pred_boxes):
        box = box.cpu().detach().numpy()
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=pred_color, facecolor='none')
        ax.add_patch(rect)

        if pred_labels is not None and pred_scores is not None:
            label = pred_labels[i]
            score = pred_scores[i]
            ax.text(x1, y1, f'{label}: {score:.2f}', bbox=dict(facecolor=pred_color, alpha=0.5))

    plt.axis('off')
    plt.savefig(os.path.join(hypar["valid_out_dir"], f'{image_name}.png'), bbox_inches='tight')
    plt.close()

    pred_boxes_array = [box.cpu().detach().numpy() for box in pred_boxes]
    npy_path = os.path.join(hypar["valid_out_bbox_dir"], f'{image_name}.npy')
    np.save(npy_path, pred_boxes_array)
    print(f"Saved predicted boxes to {npy_path}")
    # plt.show()
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
            imidx,image_name,inputs, labels = data
            
            if torch.cuda.is_available():
                inputs_v = list(input_v.cuda(hypar['gpu_id'][0]) for input_v in inputs)
                labels_v = [{k: v.cuda(hypar['gpu_id'][0]) for k, v in t.items()} for t in labels]
            else:
                inputs_v = list(input_v for input_v in inputs)
                labels_v = [{k: v for k, v in t.items()} for t in labels]

            start_inf_loss_back = time.time()
            optimizer.zero_grad()
            loss_dict = net(inputs_v, labels_v)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            # # print statistics
            running_loss += losses.item()
            del losses
            end_inf_loss_back = time.time()-start_inf_loss_back

            print(">>>"+model_path.split('/')[-1]+" - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f,  time-per-iter: %3f s, time_read: %3f" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,  time.time()-start_last, time.time()-start_last-end_inf_loss_back))
            start_last = time.time()

            if ite_num % model_save_fre == 0:  # validate every 2000 iterations
                notgood_cnt += 1
                net.eval()
                tmp_dice,  i_val, tmp_time = valid(net, valid_dataloader,hypar, epoch)
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
                    maxdice = '_'.join(tmp_dice_str)
                    torch.save(net.state_dict(), model_path+ '.pth')

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

    for i_val, data_val in enumerate(valid_dataloader):
        val_cnt = val_cnt + 1.0
        imidx,image_name,inputs_val, labels_val = data_val

        if torch.cuda.is_available():
            inputs_val_v = list(input_v.cuda(hypar['gpu_id'][0]) for input_v in inputs_val)
            labels_val_v = [{k: v.cuda(hypar['gpu_id'][0]) for k, v in t.items()} for t in labels_val]
        else:
            inputs_val_v = list(input_v for input_v in inputs_val)
            labels_val_v = [{k: v for k, v in t.items()} for t in labels_val]
        t_start = time.time()
        # if inputs_val_v.shape[1]==1:
        #     inputs_val_v=inputs_val_v.repeat(1,3,1,1)
        outputs = net(inputs_val_v)
        t_end=time.time()-t_start
        tmp_time.append(t_end)
        
        batch_mean_iou = 0.0
        batch_boxes = 0
        for target, output in zip(labels_val_v, outputs):
            output = apply_nms(output, 0.5,0.5)
            gt_boxes = target['boxes']
            pred_boxes = output['boxes']
            image = np.squeeze(io.imread(os.path.join('/'.join(hypar['val_datapath'].split('/')[:-1])+'/'+hypar['val_datapath'].split('/')[-1].split('_')[0],image_name[0],'img.jpg'))) # max = 255
            if hypar["plot_output"]:
                plot_image_with_boxes(image_name[0],image,gt_boxes,pred_boxes,output['labels'],output['scores'])
            
            for gt_box in gt_boxes:
                best_iou = 0.0
                for pred_box in pred_boxes:
                    iou = calculate_iou(gt_box.cpu().detach().numpy(), pred_box.cpu().detach().numpy())
                    if iou > best_iou:
                        best_iou = iou
                batch_mean_iou += best_iou
                batch_boxes += 1

        if batch_boxes > 0:
            batch_mean_iou /= batch_boxes
            all_mean_ious.append(batch_mean_iou)

        num_images += len(inputs_val_v)
        total_iou += batch_mean_iou
        gc.collect()
        torch.cuda.empty_cache()

        print("[validating: %5d/%5d] iou: %f, time: %f"% (i_val, val_num, batch_mean_iou,t_end))

    print('============================')
    mean_iou = np.mean(all_mean_ious) if num_images > 0 else 0
    print(f"Mean IoU: {mean_iou}")
    
    tmp_iou.append(mean_iou)


    return tmp_iou, i_val, tmp_time

def main(hypar): # model: "train", "test"
    if(hypar["mode"]=="train"):
        print("--- create training dataloader ---")
        train_dataset=myDataset(hypar['train_datapath'],[myRandomVFlip(),myRandomHFlip(),myNormalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
        hypar['train_num']=len(train_dataset)
        train_dataloader=DataLoader(train_dataset, batch_size=hypar["batch_size_train"], shuffle=True, num_workers=8, pin_memory=False,collate_fn=lambda x: tuple(zip(*x)))
    print("--- create validation dataloader ---")
    val_dataset=myDataset(hypar['val_datapath'],[myNormalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
    hypar['val_num']=len(val_dataset)
    val_dataloader=DataLoader(val_dataset, batch_size=hypar["batch_size_valid"], shuffle=False, num_workers=1, pin_memory=False,collate_fn=lambda x: tuple(zip(*x)))

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
    hypar['dataset']='INbreast-seg&det'
    hypar['finetune']='lp'
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
    input_path = f'{current_dir}/../data/{hypar["dataset"]}/Train'  # 替换为你的实际路径
    hypar['train_datapath'] = input_path+'_cache_'+str(hypar["input_size"][0])
    if not os.path.exists(hypar['train_datapath']):
        preprocess(input_path,hypar['train_datapath'],hypar["input_size"])
    input_path = f'{current_dir}/../data/{hypar["dataset"]}/Eval'  # 替换为你的实际路径
    hypar['val_datapath'] = input_path+'_cache_'+str(hypar["input_size"][0])
    if not os.path.exists(hypar['val_datapath']):
        preprocess(input_path,hypar['val_datapath'],hypar["input_size"])
    
    
    # # # # # #VersaMammo
    if hypar["mode"] == "train":
        hypar["valid_out_dir"] = "" ## for "train" model leave it as "", for "valid"("inference") mode: set it according to your local directory
        hypar["model_path"] =f"{os.path.dirname(current_dir)}/Detection/saved_model/{hypar['dataset']}/VersaMammo" ## model weights saving (or restoring) path
        hypar["restore_model"] = "" ## name of the segmentation model weights .pth for resume training process from last stop or for the inferencing
        hypar["start_ite"] = 0 ## start iteration for the training, can be changed to match the restored training process
        hypar["plot_output"]=False
    else:
        hypar["valid_out_dir"] = f"{os.path.dirname(current_dir)}/Detection/results/{hypar['dataset']}/VersaMammo"##"../DIS5K-Results-test" ## output inferenced segmentation maps into this fold
        hypar["valid_out_bbox_dir"] = f"{os.path.dirname(current_dir)}/Detection/bbox_results/{hypar['dataset']}/VersaMammo"
        hypar["restore_model"] = ""
    hypar["model"]=get_model(backbone_name="VersaMammo", checkpoint_path=f'{current_dir}/../../Sotas/VersaMammo/ENb5/ENB5_SL.pth',pretrained=True,input_size=hypar["input_size"][0])
    main(hypar=hypar)
    
    