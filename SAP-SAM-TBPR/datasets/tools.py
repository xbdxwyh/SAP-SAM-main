
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import torch

import torch.nn.functional as F


def merge_mask(mask_list,score_list):
    #mask_tensor = torch.zeros()
    mask_list = [np.array(i) for i in mask_list]
    mask_tensor = np.zeros_like(mask_list[0])
    score_best = 0
    for idx,item in enumerate(zip(mask_list,score_list)):
        mask,score = item
        if score> score_best:
            score_best = score
            mask_tensor = np.where(mask==1, np.full_like(mask,[idx+1]), mask_tensor)
            pass
        else:
            mask_tensor = np.where((mask==1)&(mask_tensor==0), np.full_like(mask,[idx+1]), mask_tensor)
            pass
    return mask_tensor


def get_patch_label(patch_mask):
    # 先统计patch的每个点数值
    unique,count=np.unique(patch_mask,return_counts=True)
    # 做排序
    sorted_id = sorted(range(count.shape[0]), key=lambda k: count[k], reverse=True)
    unq_list = [unique[i] for i in sorted_id] 
    # 默认是最多的那个
    label = unq_list[0]
    # 处理是多个，且0不为最多的情况
    if len(unq_list) > 1 and unq_list[0] == 0:
        label = unq_list[1]
    
    return label

def generate_token_class(mask_tensor,patch_size=(16,16)):
    mask_size = mask_tensor.shape
    width,height = mask_size[-2],mask_size[-1]
    if width % patch_size[0] != 0 or height % patch_size[1] != 0:
        raise ValueError("The size of MASK {} is not a multiple of patchsize {} !".format((width,height),patch_size))
    
    token_semantic_label = []
    
    for i in range(int(width // patch_size[0])):
        for j in range(int(height // patch_size[1])):
            mask_patch = mask_tensor[...,i*patch_size[0]:(i+1)*patch_size[0],j*patch_size[1]:(j+1)*patch_size[1]]
            token_semantic_label.append(get_patch_label(mask_patch))
            pass
    return token_semantic_label
    pass

