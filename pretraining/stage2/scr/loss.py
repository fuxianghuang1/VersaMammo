import torch
import torch.nn.functional as F

def contrastive_loss(fea_h_res, fea_l_res, temp=0.07):  

    # Reshape labels to ensure they are in the correct shape

    sim_img = fea_h_res @ fea_h_res.t() / temp
    sim_targets = F.softmax(sim_img, dim=1) 
    sim_predict = fea_l_res @ fea_l_res.t() / temp

    loss = -torch.sum(F.log_softmax(sim_predict, dim=1) * sim_targets, dim=1).mean()

    return loss   

def kl_loss(fea_teachert, fea_student):
    fea_teachert = F.normalize(fea_teachert, p=2, dim=-1)
    fea_student = F.normalize(fea_student, p=2, dim=-1)
    x1 = torch.mm(fea_teachert, fea_teachert.transpose(0, 1)) 
    x2 = torch.mm(fea_student, fea_student.transpose(0, 1))
    log_soft_x1 = F.log_softmax(x1, dim=1)
    soft_x2 = F.softmax(x2, dim=1)
    kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')
    return kl

