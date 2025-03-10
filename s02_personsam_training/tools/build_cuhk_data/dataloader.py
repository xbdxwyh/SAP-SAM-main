# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""

from torchvision import transforms
from PIL import Image
import torch
from .dataset import CUHKPEDEDataset, CUHKPEDE_img_dateset, CUHKPEDE_txt_dateset


def get_dataloader(opt):
    """
    tranforms the image, downloads the image with the id by data.DataLoader
    """

    if opt.mode == 'train':
        dataset = CUHKPEDEDataset(opt)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                                 shuffle=True, drop_last=True, num_workers=1)
        print('{}-{} has {} pohtos'.format(opt.dataset, opt.mode, len(dataset)))

        return dataloader

    else:
        img_dataset = CUHKPEDE_img_dateset(opt)

        img_dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=opt.batch_size,
                                                 shuffle=False, drop_last=False, num_workers=1)

        txt_dataset = CUHKPEDE_txt_dateset(opt)

        txt_dataloader = torch.utils.data.DataLoader(txt_dataset, batch_size=opt.batch_size,
                                                 shuffle=False, drop_last=False, num_workers=1)

        print('{}-{} has {} pohtos, {} text'.format(opt.dataset, opt.mode, len(img_dataset), len(txt_dataset)))

        return img_dataloader, txt_dataloader