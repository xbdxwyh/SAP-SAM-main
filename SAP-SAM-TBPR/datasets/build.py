import logging
import torch
import torchvision.transforms as T
import os, sys
sys.path.append("..")

from torch.utils.data import DataLoader
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextDataset, ImageTextMLMDataset

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid

from . import dataAugment 

from .aug import (
    random_rotate,
    random_h_flip,
    random_crop,
    random_erasing,
    random_adjust_sharpness,
    random_affine,
    random_auto_contrast,
    random_color_jittor,
    random_equalize,
    random_gaussian_blur,
    random_grayscale,
    random_invert,
    random_posterize,
    random_apply,
    augmix
)

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        aug_list = [
            #T.Resize((height, width)),# OK
            T.RandomHorizontalFlip(0.5),# OK
            T.RandomGrayscale(0.5), # OK
            T.ColorJitter(brightness=.15, hue=.1), # OK
            T.GaussianBlur(kernel_size=(3, 5), sigma=(0.05, 1.5)),  # OK
            #T.Pad(10), # OK
            T.RandomInvert(), # OK
            T.RandomPosterize(bits=3), # OK
            T.RandomAdjustSharpness(sharpness_factor=4),# OK
            T.RandomAutocontrast(), # OK
            #T.TrivialAugmentWide(),
            T.AugMix(severity=3),# OK, 版本原因
            T.RandomRotation(25),# OK
            #T.ElasticTransform(alpha=15.0),
            #T.RandomCrop((height, width)),# OK
            T.RandomAffine(degrees=(5, 15),translate=(0,0),shear=10), # OK
            T.RandomEqualize(), # OK
            # T.ToTensor(),
            # T.Normalize(mean=mean, std=std),
            # T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ]
        transform = T.Compose([
            #T.Pad(10), # OK
            T.Resize((height, width)),
            T.RandomCrop((height, width)),
            #T.RandomApply(transforms=aug_list,p=0.45),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform

def build_pair_transforms_class(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform
    # transform for training
    if aug:
        aug_list = [
            #T.Resize((height, width)),# OK
            dataAugment.ImageMaskRandomHFlip(p=0.5),# OK
            dataAugment.ImageMaskRandomGrayscale(p=0.5), # OK
            dataAugment.ImageMaskColorJitter(brightness=.15, hue=.1), # OK
            dataAugment.ImageMaskGaussianBlur(kernel_size=(3, 5), sigma=(0.02, 1.5)),  # OK
            # #T.Pad(10), # OK
            dataAugment.ImageMaskRandomInvert(), # OK
            dataAugment.ImageMaskRandomPosterize(bits=3), # OK
            dataAugment.ImageMaskRandomAdjustSharpness(sharpness_factor=4),# OK
            dataAugment.ImageMaskRandomAutocontrast(), # OK
            # #T.TrivialAugmentWide(),
            dataAugment.ImageMaskAugMix(severity=3),# OK, 版本原因
            dataAugment.ImageMaskRandomRotation(25),# OK
            #T.ElasticTransform(alpha=15.0),
            #T.RandomCrop((height, width)),# OK
            dataAugment.ImageMaskRandomAffine(degrees=(5, 15),translate=(0,0),shear=10), # OK
            dataAugment.ImageMaskRandomEqualize(), # OK
            # T.ToTensor(),
            # T.Normalize(mean=mean, std=std),
            # T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ]
        tools_list = [
            (width,height),
            mean,
            std,
            [
                dataAugment.ImageMaskRandomApply(aug_list,p=0.45), 
                dataAugment.ImageMaskRandomCrop((height, width)),
                dataAugment.ImageMaskRandomErasing(scale=(0.02, 0.4), value=mean)
            ]
        ]
    else:
        tools_list = [(width,height),mean,std,[random_h_flip]]
    return tools_list


def build_pair_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform
    # transform for training
    if aug:
        random_transforms = [
            random_h_flip,
            random_grayscale,
            random_color_jittor,
            random_gaussian_blur,
            random_invert,
            random_posterize,
            random_adjust_sharpness,
            random_auto_contrast,
            augmix,
            random_rotate,
            random_affine,
            random_equalize
        ]
        tools_list = [
            (width,height),
            mean,
            std,
            [random_transforms,random_apply,random_crop,random_erasing]
        ]
    else:
        tools_list = [(width,height),mean,std,[random_h_flip]]
    return tools_list


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    #print(keys)
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict

def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir,part_seg=args.part_seg)
    num_classes = len(dataset.train_id_container)
    
    if args.training:
        if args.part_seg:
            train_transforms = build_pair_transforms_class(
                img_size=args.img_size,
                aug=args.img_aug,
                is_train=True
            )
        else:
            print(args.img_size,type(args.img_size),tuple(args.img_size))
            train_transforms = build_transforms(
                img_size=args.img_size,
                aug=args.img_aug,
                is_train=True
            )
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)

        if args.MLM:
            train_set = ImageTextMLMDataset(
                dataset.train,
                train_transforms,
                text_length=args.text_length,
                part_seg = args.part_seg,
                using_mim = "mim" in args.loss_names,
                return_attr_tokens=args.return_attr_tokens,
                part_mask_prob=args.part_mask_prob,
                mask_prob=args.mask_prob
            )
        else:
            train_set = ImageTextDataset(
                dataset.train,
                train_transforms,
                text_length=args.text_length,
                part_seg = args.part_seg
            )

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                # TODO wait to fix bugs
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)

            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'],
                                  text_length=args.text_length)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # build dataloader for testing
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        return test_img_loader, test_txt_loader, num_classes
