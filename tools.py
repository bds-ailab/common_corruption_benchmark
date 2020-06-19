# Copyright 2020 BULL SAS All rights reserved #

import torch

def update_topk_acc(acc,y_found,target_labels,nb_iter,k):
    _,l_found = torch.topk(y_found,k)
    target_labels = target_labels.unsqueeze(1).repeat(1,k)
    acc = acc + torch.eq(l_found,target_labels).float().mean()*k/nb_iter
    return acc
