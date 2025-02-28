# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
from .read_write_data import read_dict
from .transforms import transforms
#import cv2
# import torchvision.transforms.functional as F
import random
import re
#from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import CLIPTokenizer,BertTokenizer
from PIL import ImageStat
import copy

def fliplr(img, dim):
    """
    flip horizontal
    :param img:
    :return:
    """
    inv_idx = torch.arange(img.size(dim) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(dim, inv_idx)
    return img_flip

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

def convert_examples_to_features_token(examples, seq_length, tokenizer):
    examples = [i.text_a for i in examples]
    tokenized_examples = tokenizer(examples,padding="max_length",max_length=seq_length,truncation=True)
    #print(tokenized_examples['attention_mask'])
    token_length = [sum(i) for i in tokenized_examples['attention_mask']]
    return tokenized_examples['input_ids'][0], tokenized_examples['attention_mask'][0],token_length[0]


def load_data_transformers(resize_reso, crop_reso, swap_num=(12, 4)):
    data_transforms = {
       	'swap': transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize((384, 128), Image.BICUBIC),  # interpolation
            # transforms.RandomRotation(degrees=15),
            # transforms.RandomCrop((crop_reso[0], crop_reso[1])),
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
        'common_aug': transforms.Compose([
            # transforms.Resize((resize_reso[0], resize_reso[1]),Image.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((384, 128), Image.BICUBIC),  # interpolation
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso[0], crop_reso[1])),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'train_totensor': transforms.Compose([
            # transforms.Resize((crop_reso[0], crop_reso[1]),Image.BICUBIC),
            # ImageNetPolicy(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ]),
        'test_totensor': transforms.Compose([
            transforms.Resize((crop_reso[0], crop_reso[1]),Image.BICUBIC),
            transforms.CenterCrop((crop_reso[0], crop_reso[1])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'None': None,
    }
    return data_transforms

class CUHKPEDEDataset(data.Dataset):
    def __init__(self, opt):

        self.opt = opt
        self.flip_flag = (self.opt.mode == 'train')

        data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))

        self.img_path = [os.path.join(opt.dataroot, img_path) for img_path in data_save['img_path']]

        self.label = data_save['id']

        self.caption_code = data_save['lstm_caption_id']

        self.same_id_index = data_save['same_id_index']

        self.caption = data_save['captions']

        self.num_data = len(self.img_path)

        if opt.wordtype == "bert" or opt.wordtype == "BERT" or opt.wordtype =="Bert":
            #self.tokenizer = BertTokenizer.from_pretrained('F:\\preTrainedModels\\bert-base-uncased\\vocab.txt', do_lower_case=True)
            self.tokenizer = BertTokenizer.from_pretrained(opt.pkl_root)
        elif opt.wordtype == "clip" or opt.wordtype == "CLIP":
            self.tokenizer = CLIPTokenizer.from_pretrained(opt.pkl_root)

        self.transformers = load_data_transformers([384,128], [384,128], [4,6])

        self.swap_size = [4,6]
    def crop_image(self, image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """
        image = Image.open(self.img_path[index])
        img_unswaps = self.transformers['common_aug'](image)
        img_unswaps = self.transformers["train_totensor"](img_unswaps)
        label = torch.from_numpy(np.array([self.label[index]],dtype='int32')).long()

        phrase = self.caption[index]
        examples = read_examples(phrase, index)

        caption_input_ids,caption_attention_mask,caption_length = convert_examples_to_features_token(
            examples=examples, 
            seq_length=self.opt.caption_length_max, 
            tokenizer=self.tokenizer
        )
        # same_id_index = np.random.randint(len(self.same_id_index[index]))
        # same_id_index = self.same_id_index[index][same_id_index]
        # phrase = self.caption[same_id_index]
        # examples = read_examples(phrase, index)

        # same_caption_input_ids,same_caption_attention_mask,same_caption_length = convert_examples_to_features_token(
        #     examples=examples, 
        #     seq_length=self.opt.caption_length_max, 
        #     tokenizer=self.tokenizer
        # )

        return img_unswaps, label, \
            np.array(caption_input_ids,dtype=int), np.array(caption_attention_mask,dtype=int), np.array(caption_length,dtype=int), \
            np.array(caption_input_ids,dtype=int), np.array(caption_attention_mask,dtype=int), np.array(caption_length,dtype=int)
        
    def get_text(self, index):
        """
        :param index:
        :return: image and its label
        """
        
        label = torch.from_numpy(np.array([self.label[index]],dtype='int32')).long()

        phrase = self.caption[index]
        examples = read_examples(phrase, index)

        caption_input_ids,caption_attention_mask,caption_length = convert_examples_to_features_token(
            examples=examples, 
            seq_length=self.opt.caption_length_max, 
            tokenizer=self.tokenizer
        )

        return None, label, \
            np.array(caption_input_ids,dtype=int), np.array(caption_attention_mask,dtype=int), np.array(caption_length,dtype=int), \
            np.array(caption_input_ids,dtype=int), np.array(caption_attention_mask,dtype=int), np.array(caption_length,dtype=int)
            #None,None,None,
            # np.array(same_caption_input_ids,dtype=int), np.array(same_caption_attention_mask,dtype=int), np.array(same_caption_length,dtype=int)

    def __len__(self):
        return self.num_data


class CUHKPEDE_img_dateset(data.Dataset):
    def __init__(self, opt):

        self.opt = opt
        if opt.mode=='train':
            path = opt.dataroot+'processed_data/train_save.pkl'
        elif opt.mode=='test':
            path = opt.dataroot+'processed_data/test_save.pkl'

        data_save = read_dict(path)

        self.img_path = [os.path.join(opt.dataroot, img_path) for img_path in data_save['img_path']]

        self.label = data_save['id']

        self.num_data = len(self.img_path)

        self.transformers = load_data_transformers([384, 128], [384, 128], [12, 4])

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        image = Image.open(self.img_path[index])
        image_path = self.img_path[index]
        #raw_image = cv2.imread(image_path)
        # raw_image = cv2.resize(raw_image, (128, 384), interpolation=cv2.INTER_CUBIC)
        # # image = self.transform(image)
        image = self.transformers["test_totensor"](image)

        label = torch.from_numpy(np.array([self.label[index]])).long()

        return image, label

    def __len__(self):
        return self.num_data


class CUHKPEDE_txt_dateset(data.Dataset):
    def __init__(self, opt):

        self.opt = opt

        data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))

        self.label = data_save['caption_label']
        self.caption_code = data_save['lstm_caption_id']
        self.caption = data_save['captions']
        self.num_data = len(self.caption_code)
        self.caption_matching_img_index = data_save['caption_matching_img_index']

        if opt.wordtype == "bert" or opt.wordtype == "BERT" or opt.wordtype =="Bert":
            self.tokenizer = BertTokenizer.from_pretrained(opt.pkl_root)
        elif opt.wordtype == "clip" or opt.wordtype == "CLIP":
            self.tokenizer = CLIPTokenizer.from_pretrained(opt.pkl_root)

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        label = torch.from_numpy(np.array([self.label[index]])).long()

        phrase = self.caption[index]
        examples = read_examples(phrase, index)
        caption_matching_img_index = self.caption_matching_img_index[index]

        caption_input_ids,caption_attention_mask,caption_length = convert_examples_to_features_token(
            examples=examples, 
            seq_length=self.opt.caption_length_max, 
            tokenizer=self.tokenizer
        )
        
        return label, np.array(caption_input_ids,dtype=int), np.array(caption_length,dtype=int),np.array(caption_attention_mask,dtype=int),caption_matching_img_index

    def __len__(self):
        return self.num_data





