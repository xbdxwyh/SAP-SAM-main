# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json
import os
from .read_write_data import read_dict
from .transforms import transforms
#import cv2
# import torchvision.transforms.functional as F
import random
import re
#from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import CLIPTokenizer,BertTokenizer,SamImageProcessor
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
    return tokenized_examples['input_ids'][0], tokenized_examples['attention_mask'][0],tokenized_examples['token_type_ids'][0],token_length[0]


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
    def __init__(self, opt, sam_processor):

        self.opt = opt
        self.flip_flag = (self.opt.mode == 'train')

        # read loaded data
        data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))
        
        # read processed attribute data
        # 将所有的文本进行分割后得到的新数据json，进行读取
        with open(os.path.join(opt.dataroot, "data_final_68120.json"),"r+") as f:
            data = json.load(f)
        
        # 读取使用SAM分割后的所有数据集名称以及置信度
        with open(os.path.join(opt.dataroot, "sam_score_dict_cuhk.json"),"r+") as f:
            data_seg = json.load(f)
    
        img_path = data_save['img_path']
        label = data_save['id']
        caption_code = data_save['lstm_caption_id']
        same_id_index = data_save['same_id_index']
        caption = data_save['captions']

        img_path_list = []
        label_list = []
        caption_code_list = []
        same_id_index_list = []
        caption_list = []
        attribute_list = []
        seg_name_list = []
        seg_score_list = []

        # 处理每一条json，并设置进数据集中
        for data_id,item in enumerate(data):
            idx = item['idx']
            img_path_list.append(os.path.join(opt.dataroot, img_path[idx]))
            label_list.append(label[idx])
            caption_code_list.append(caption_code[idx])
            same_id_index_list.append(same_id_index[idx])
            caption_list.append(caption[idx])
            attribute_list.append(item['attribute'])
            seg_name_temp = ["_".join([str(data_id)] + item['name'][:-len(".png")].split("/")+[str(i)]) for i in range(len(item['attribute']))]
            seg_score_list.append([data_seg[k] for k in seg_name_temp])
            seg_name_list.append(seg_name_temp)
            assert img_path[idx][5:] == item['name']


        self.img_path = img_path_list
        self.attribute_list = attribute_list
        self.label = label_list
        self.caption_code = caption_code_list
        self.same_id_index = same_id_index_list
        self.caption = caption_list
        self.seg_name = seg_name_list
        self.seg_score = seg_score_list

        self.num_data = len(self.img_path)

        if opt.wordtype == "bert" or opt.wordtype == "BERT" or opt.wordtype =="Bert":
            #self.tokenizer = BertTokenizer.from_pretrained('F:\\preTrainedModels\\bert-base-uncased\\vocab.txt', do_lower_case=True)
            self.tokenizer = BertTokenizer.from_pretrained(opt.pkl_root)
        elif opt.wordtype == "clip" or opt.wordtype == "CLIP":
            self.tokenizer = CLIPTokenizer.from_pretrained(opt.pkl_root)

        #self.transformers = load_data_transformers([384,128], [384,128], [4,6])

        self.swap_size = [4,6]

        self.sam_processor = sam_processor

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

        image_inputs = self.sam_processor(image, return_tensors="pt")
        image_inputs = {k:v.squeeze(0) for k,v in image_inputs.items()}

        # name = self.img_path[index][len(self.opt.dataroot+"imgs/"):]

        # img_unswaps = self.transformers['common_aug'](image)
        # img_unswaps = self.transformers["train_totensor"](img_unswaps)
        label = torch.from_numpy(np.array([self.label[index]],dtype='int32')).long()

        phrase = self.caption[index]
        examples = read_examples(phrase, index)

        caption_input_ids,caption_attention_mask,caption_token_type_ids,caption_length = convert_examples_to_features_token(
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
        attribute = self.attribute_list[index]

        while True:
            if isinstance(attribute[0], list):
                attribute = attribute[0]
            else:
                break
            pass

        if len(attribute) >= self.opt.attr_num:
            attribute = random.sample(attribute,self.opt.attr_num)
        else:
            attribute = attribute+[random.sample(attribute,1)[0] for i in range(self.opt.attr_num-len(attribute))]
        
        tokenized_attribute = self.tokenizer(
            attribute,
            padding="max_length",
            max_length=self.opt.attr_len,
            truncation=True,
            return_tensors="pt"
        )

        processed_item = {
            "label": torch.LongTensor(label),
            "input_ids": torch.LongTensor(caption_input_ids),
            "attention_mask": torch.LongTensor(caption_attention_mask),
            "token_type_ids": torch.LongTensor(caption_token_type_ids),
            #"length": torch.LongTensor(caption_length),
            "attribute_input_ids":tokenized_attribute['input_ids'],
            "attribute_attention_mask":tokenized_attribute['attention_mask'],
            "attribute_token_type_ids":tokenized_attribute['token_type_ids'],
        }

        for k,v in image_inputs.items():
            processed_item[k] = v

        return processed_item

        # return img_unswaps, label, \
        #     np.array(caption_input_ids,dtype=int), np.array(caption_attention_mask,dtype=int), np.array(caption_length,dtype=int), \
        #     tokenized_attribute,\
        #     np.array(caption_input_ids,dtype=int), np.array(caption_attention_mask,dtype=int), np.array(caption_length,dtype=int)
        
    def get_origin_data(self, index, return_image=False, processed=False, return_seg=False):
        """
        :param index:
        :return: image and its label
        """
        
        label = torch.from_numpy(np.array([self.label[index]],dtype='int32')).long()
        if return_image:
            image = Image.open(self.img_path[index])
            if processed:
                image = self.sam_processor(image, return_tensors="pt")
                image = {k:v.squeeze(0) for k,v in image.items()}
                pass
            pass
        else:
            image = None

        phrase = self.caption[index]
        examples = read_examples(phrase, index)
        examples = [i.text_a for i in examples]
        if processed:
            examples = self.tokenizer(
                examples,
                padding="max_length",
                max_length=self.opt.caption_length_max,
                truncation=True
            )

        attribute = self.attribute_list[index]

        seg_img = None
        seg_score = None
        if return_seg:
            seg_name = self.seg_name[index]
            seg_score = self.seg_score[index]
            seg_img = [Image.open(os.path.join(self.opt.dataroot,"segs",k+".png")) for k in seg_name]
            pass

        while True:
            if isinstance(attribute[0], list):
                attribute = attribute[0]
            else:
                break
            pass

        return image, label, examples, attribute, seg_img, seg_score
            #None,None,None,
            # np.array(same_caption_input_ids,dtype=int), np.array(same_caption_attention_mask,dtype=int), np.array(same_caption_length,dtype=int)

    def __len__(self):
        return self.num_data


class CUHKPEDE_img_dateset(data.Dataset):
    def __init__(self, opt, sam_processor):

        self.opt = opt
        if opt.mode=='train':
            path = opt.dataroot+'processed_data/train_save.pkl'
        elif opt.mode=='test':
            path = opt.dataroot+'processed_data/test_save.pkl'

        data_save = read_dict(path)

        self.sam_processor = sam_processor

        self.img_path = [os.path.join(opt.dataroot, img_path) for img_path in data_save['img_path']]

        self.label = data_save['id']

        self.num_data = len(self.img_path)

        #self.transformers = load_data_transformers([384, 128], [384, 128], [12, 4])

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        image = Image.open(self.img_path[index])

        image_inputs = self.sam_processor(image, return_tensors="pt")
        image_inputs = {k:v.squeeze(0) for k,v in image_inputs.items()}

        label = torch.from_numpy(np.array([self.label[index]])).long()
        image_inputs['label'] = torch.LongTensor(label)

        return image_inputs

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

        caption_input_ids,caption_attention_mask,caption_token_type_ids,caption_length = convert_examples_to_features_token(
            examples=examples, 
            seq_length=self.opt.caption_length_max, 
            tokenizer=self.tokenizer
        )

        processed_item = {
            "label": torch.LongTensor(label),
            "input_ids": torch.LongTensor(caption_input_ids),
            "attention_mask": torch.LongTensor(caption_attention_mask),
            "token_type_ids": torch.LongTensor(caption_token_type_ids),
        }
        
        return processed_item,caption_matching_img_index

    def __len__(self):
        return self.num_data





class CUHKPEDEDatasetOld(data.Dataset):
    def __init__(self, opt, sam_processor):

        self.opt = opt
        self.flip_flag = (self.opt.mode == 'train')

        # read loaded data
        data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))
        
        # read processed attribute data
        # 将所有的文本进行分割后得到的新数据json，进行读取
        with open(os.path.join(opt.dataroot, "data_final_68120.json"),"r+") as f:
            data = json.load(f)
    
        img_path = data_save['img_path']
        label = data_save['id']
        caption_code = data_save['lstm_caption_id']
        same_id_index = data_save['same_id_index']
        caption = data_save['captions']

        img_path_list = []
        label_list = []
        caption_code_list = []
        same_id_index_list = []
        caption_list = []
        attribute_list = []

        # 处理每一条json，并设置进数据集中
        for item in data:
            idx = item['idx']
            img_path_list.append(os.path.join(opt.dataroot, img_path[idx]))
            label_list.append(label[idx])
            caption_code_list.append(caption_code[idx])
            same_id_index_list.append(same_id_index[idx])
            caption_list.append(caption[idx])
            attribute_list.append(item['attribute'])
            assert img_path[idx][5:] == item['name']


        self.img_path = img_path_list
        self.attribute_list = attribute_list
        self.label = label_list
        self.caption_code = caption_code_list
        self.same_id_index = same_id_index_list
        self.caption = caption_list

        self.num_data = len(self.img_path)

        if opt.wordtype == "bert" or opt.wordtype == "BERT" or opt.wordtype =="Bert":
            #self.tokenizer = BertTokenizer.from_pretrained('F:\\preTrainedModels\\bert-base-uncased\\vocab.txt', do_lower_case=True)
            self.tokenizer = BertTokenizer.from_pretrained(opt.pkl_root)
        elif opt.wordtype == "clip" or opt.wordtype == "CLIP":
            self.tokenizer = CLIPTokenizer.from_pretrained(opt.pkl_root)

        #self.transformers = load_data_transformers([384,128], [384,128], [4,6])

        self.swap_size = [4,6]

        self.sam_processor = sam_processor

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

        image_inputs = self.sam_processor(image, return_tensors="pt")
        image_inputs = {k:v.squeeze(0) for k,v in image_inputs.items()}

        # name = self.img_path[index][len(self.opt.dataroot+"imgs/"):]

        # img_unswaps = self.transformers['common_aug'](image)
        # img_unswaps = self.transformers["train_totensor"](img_unswaps)
        label = torch.from_numpy(np.array([self.label[index]],dtype='int32')).long()

        phrase = self.caption[index]
        examples = read_examples(phrase, index)

        caption_input_ids,caption_attention_mask,caption_token_type_ids,caption_length = convert_examples_to_features_token(
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
        attribute = self.attribute_list[index]

        while True:
            if isinstance(attribute[0], list):
                attribute = attribute[0]
            else:
                break
            pass

        if len(attribute) >= self.opt.attr_num:
            attribute = random.sample(attribute,self.opt.attr_num)
        else:
            attribute = attribute+[random.sample(attribute,1)[0] for i in range(self.opt.attr_num-len(attribute))]
        
        tokenized_attribute = self.tokenizer(
            attribute,
            padding="max_length",
            max_length=self.opt.attr_len,
            truncation=True,
            return_tensors="pt"
        )

        processed_item = {
            "label": torch.LongTensor(label),
            "input_ids": torch.LongTensor(caption_input_ids),
            "attention_mask": torch.LongTensor(caption_attention_mask),
            "token_type_ids": torch.LongTensor(caption_token_type_ids),
            #"length": torch.LongTensor(caption_length),
            "attribute_input_ids":tokenized_attribute['input_ids'],
            "attribute_attention_mask":tokenized_attribute['attention_mask'],
            "attribute_token_type_ids":tokenized_attribute['token_type_ids'],
        }

        for k,v in image_inputs.items():
            processed_item[k] = v

        return processed_item

        # return img_unswaps, label, \
        #     np.array(caption_input_ids,dtype=int), np.array(caption_attention_mask,dtype=int), np.array(caption_length,dtype=int), \
        #     tokenized_attribute,\
        #     np.array(caption_input_ids,dtype=int), np.array(caption_attention_mask,dtype=int), np.array(caption_length,dtype=int)
        
    def get_origin_data(self, index, return_image=False, processed=False):
        """
        :param index:
        :return: image and its label
        """
        
        label = torch.from_numpy(np.array([self.label[index]],dtype='int32')).long()
        if return_image:
            image = Image.open(self.img_path[index])
            if processed:
                image = self.sam_processor(image, return_tensors="pt")
                image = {k:v.squeeze(0) for k,v in image.items()}
                pass
            pass
        else:
            image = None

        phrase = self.caption[index]
        examples = read_examples(phrase, index)
        examples = [i.text_a for i in examples]
        if processed:
            examples = self.tokenizer(
                examples,
                padding="max_length",
                max_length=self.opt.caption_length_max,
                truncation=True
            )

        attribute = self.attribute_list[index]

        while True:
            if isinstance(attribute[0], list):
                attribute = attribute[0]
            else:
                break
            pass

        return image, label, examples, attribute
            #None,None,None,
            # np.array(same_caption_input_ids,dtype=int), np.array(same_caption_attention_mask,dtype=int), np.array(same_caption_length,dtype=int)

    def __len__(self):
        return self.num_data
    

class TBPRBaseDataset(data.Dataset):
    def __init__(self, opt, sam_processor):

        self.opt = opt
        self.flip_flag = (self.opt.mode == 'train')

        # read loaded data
        data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))
        
        # read processed attribute data
        # 将所有的文本进行分割后得到的新数据json，进行读取
        with open(os.path.join(opt.dataroot, "data_final.json"),"r+") as f:
            data = json.load(f)
        
        img_path = data_save['img_path']
        label = data_save['id']
        caption_code = data_save['lstm_caption_id']
        same_id_index = data_save['same_id_index']
        caption = data_save['captions']

        img_path_list = []
        label_list = []
        caption_code_list = []
        same_id_index_list = []
        caption_list = []
        attribute_list = []

        # 处理每一条json，并设置进数据集中
        for data_id,item in enumerate(data):
            idx = item['idx']
            img_path_list.append(os.path.join(opt.dataroot, img_path[idx]))
            label_list.append(label[idx])
            caption_code_list.append(caption_code[idx])
            same_id_index_list.append(same_id_index[idx])
            caption_list.append(caption[idx])
            attribute_list.append(item['attribute'])
            assert img_path[idx][5:] == item['name']

        self.img_path = img_path_list
        self.attribute_list = attribute_list
        self.label = label_list
        self.caption_code = caption_code_list
        self.same_id_index = same_id_index_list
        self.caption = caption_list

        self.num_data = len(self.img_path)

        if opt.wordtype == "bert" or opt.wordtype == "BERT" or opt.wordtype =="Bert":
            #self.tokenizer = BertTokenizer.from_pretrained('F:\\preTrainedModels\\bert-base-uncased\\vocab.txt', do_lower_case=True)
            self.tokenizer = BertTokenizer.from_pretrained(opt.pkl_root)
        elif opt.wordtype == "clip" or opt.wordtype == "CLIP":
            self.tokenizer = CLIPTokenizer.from_pretrained(opt.pkl_root)

        #self.transformers = load_data_transformers([384,128], [384,128], [4,6])

        self.swap_size = [4,6]

        self.sam_processor = sam_processor

    def crop_image(self, image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list


    def get_origin_data(self, index, return_image=False, processed=False, return_seg=False):
        """
        :param index:
        :return: image and its label
        """
        
        label = torch.from_numpy(np.array([self.label[index]],dtype='int32')).long()
        if return_image:
            image = Image.open(self.img_path[index])
            if processed:
                image = self.sam_processor(image, return_tensors="pt")
                image = {k:v.squeeze(0) for k,v in image.items()}
                pass
            pass
        else:
            image = None

        phrase = self.caption[index]
        examples = read_examples(phrase, index)
        examples = [i.text_a for i in examples]
        if processed:
            examples = self.tokenizer(
                examples,
                padding="max_length",
                max_length=self.opt.caption_length_max,
                truncation=True
            )

        attribute = self.attribute_list[index]

        seg_img = None
        seg_score = None
        if return_seg:
            seg_name = self.seg_name[index]
            seg_score = self.seg_score[index]
            seg_img = [Image.open(os.path.join(self.opt.dataroot,"segs",k+".png")) for k in seg_name]
            pass

        while True:
            if isinstance(attribute[0], list):
                attribute = attribute[0]
            else:
                break
            pass

        return image, label, examples, attribute, seg_img, seg_score
            #None,None,None,
            # np.array(same_caption_input_ids,dtype=int), np.array(same_caption_attention_mask,dtype=int), np.array(same_caption_length,dtype=int)

    def __len__(self):
        return self.num_data
    