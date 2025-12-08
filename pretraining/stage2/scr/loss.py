import torch
import torch.nn.functional as F

def contrastive_loss(fea_global, fea_local, birads_labels, density_labels, alpha=0.5, temp=0.07):  

    # Reshape labels to ensure they are in the correct shape
    birads_labels = birads_labels.view(-1, 1)
    density_labels = density_labels.view(-1, 1)

    sim_img = fea_global @ fea_global.t() / temp
 
    pos_birads = torch.eq(birads_labels, birads_labels.t()).float()  
    pos_density = torch.eq(density_labels, density_labels.t()).float()  

    sim_targets_birads = pos_birads / pos_birads.sum(1, keepdim=True)  
    sim_targets_density = pos_density / pos_density.sum(1, keepdim=True)  

    sim_targets = alpha * F.softmax(sim_img, dim=1) + (1 - alpha) * (sim_targets_birads + sim_targets_density) / 2
    sim_predict = fea_local @ fea_local.t() / temp

    loss = -torch.sum(F.log_softmax(sim_predict, dim=1) * sim_targets, dim=1).mean()

    return loss  

def kl_loss(fea_global, fea_local):
    fea_global = F.normalize(fea_global, p=2, dim=-1)
    fea_local = F.normalize(fea_local, p=2, dim=-1)
    x1 = torch.mm(fea_global, fea_global.transpose(0, 1)) 
    x2 = torch.mm(fea_local, fea_local.transpose(0, 1))

    log_soft_x1 = F.log_softmax(x1, dim=1)
    soft_x2 = F.softmax(x2, dim=1)
    kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')
    return kl

