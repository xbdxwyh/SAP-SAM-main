from typing import List
from torch.utils.data import Dataset
import os.path as op
import logging
import torch
import sys
sys.path.append("..")
from PIL import Image, ImageFile

import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        
        vocab.pop(-1) # remove last one in vocab(jekyll) to keep vocab_size unchanged
        vocab.extend(['<|mask|>', '<|startoftext|>', '<|endoftext|>']) # vocab_size 49408
        # vocab.extend(['<|startoftext|>', '<|endoftext|>']) # vocab_size 49408
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|mask|>': '<|mask|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|mask\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

# from utils.iotools import read_image
# from utils.simple_tokenizer import SimpleTokenizer

import os.path as osp
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

from prettytable import PrettyTable
import random
import regex as re
import copy
from PIL import Image

import torchvision.transforms.functional as F

import numpy as np

from .tools import merge_mask,generate_token_class

import torchvision.transforms.functional as tf



class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(
            self,
            dataset,
            transform=None,
            text_length: int = 77,
            truncate: bool = True,
            part_seg: bool = False,
            max_part_num: int = 6
            ):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.part_seg = part_seg
        self.max_part_num = max_part_num

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.part_seg:
            img_size,mean,std,function_list = self.transform

            pid, image_id, img_path, caption,attribute,seg_img_name,seg_img_score = self.dataset[index]
            
            # 处理一些异常情况
            while True:
                if isinstance(attribute[0], list):
                    attribute = attribute[0]
                else:
                    break
                pass
            
            img = read_image(img_path).resize(img_size)
            img_2 = read_image(img_path).resize(img_size)
            seg_img = [Image.open(img_path) for img_path in seg_img_name]
            seg_img_score = [i['score'] for i in seg_img_score]
            part_num = len(seg_img)
            if part_num >= self.max_part_num:
                seg_img = seg_img[:self.max_part_num]
                seg_img_score = seg_img_score[:self.max_part_num]
                attribute = attribute[:self.max_part_num]
                part_num = self.max_part_num
            else:
                attribute = attribute + [attribute[0] for i in range(self.max_part_num-part_num)]
            
            merged_mask = Image.fromarray(merge_mask(seg_img,seg_img_score)).resize(img_size)
            merged_mask_2 = Image.fromarray(merge_mask(seg_img,seg_img_score)).resize(img_size)

            for func in function_list:
                img,merged_mask = func(img,merged_mask)
                img_2,merged_mask_2 = func(img_2,merged_mask_2)

            if not isinstance(img,Image.Image):
                merged_mask = (merged_mask*255).long().numpy()
            else:
                img = tf.to_tensor(img)
            
            if not isinstance(img_2,Image.Image):
                merged_mask_2 = (merged_mask_2*255).long().numpy()
            else:
                img_2 = tf.to_tensor(img_2)
            
            img_norm = F.normalize(img, mean, std, False)
            img_2_norm = F.normalize(img_2, mean, std, False)
            
            label_list = generate_token_class(np.array(merged_mask))
            label = torch.LongTensor(label_list)

            label_list_2 = generate_token_class(np.array(merged_mask_2))
            label_2 = torch.LongTensor(label_list_2)

            tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
            
            attribute_tokens = torch.stack([tokenize(atr, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate) for atr in attribute],dim=0)
            
            ret = {
                'pids': pid,
                'image_ids': image_id,
                'images': img_norm,
                'images_2': img_2_norm,
                'images_origin': img,
                'caption_ids': tokens,
                'label':label,
                'label_2':label_2,
                'attribute':attribute_tokens,
                'part_num':part_num
            }

            return ret
        else:
            pid, image_id, img_path, caption = self.dataset[index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)

            tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

            ret = {
                'pids': pid,
                'image_ids': image_id,
                'images': img,
                'caption_ids': tokens,
            }

            return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


def find_subsequence_index_with_error(sentence, part, threshold=2):
    """
    Args:
        sentence: 长序列。
        part: 子序列。
        threshold: 允许的错误个数。

    Returns:
        子序列在长序列中的位置列表。
    """
    if part == []:
        return [(-1,-1)]
    index = []
    for i in range(len(sentence)):
        if sentence[i] == part[0]:
            match = True
            error_count = 0
            for j in range(1, len(part)):
                index_j = min(i + j, len(sentence) - 1)
                if sentence[index_j] != part[j]:
                    error_count += 1
                    if error_count > threshold:
                        match = False
                        break
            if match:
                index.append((i, i + len(part) - 1))
    if index == []:
        return [(-1,-1)]
    return index


class ImageTextMLMDataset(Dataset):
    def __init__(
            self,
            dataset,
            transform=None,
            text_length: int = 77,
            truncate: bool = True,
            part_seg: bool = False,
            max_part_num: int = 6,
            using_mim: bool = False,
            return_attr_tokens: bool = False,
            part_mask_prob: float = 0.5,
            mask_prob: float = 0.15
            ):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.part_seg = part_seg
        self.max_part_num = max_part_num
        self.using_mim = using_mim
        self.return_attr_tokens = return_attr_tokens
        self.part_mask_prob = part_mask_prob
        self.mask_prob = mask_prob

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.part_seg:
            img_size,mean,std,function_list = self.transform
            # random_transforms = function_list[0]
            # random_apply = function_list[1]
            # function_list = function_list[2:]

            pid, image_id, img_path, caption,attribute,seg_img_name,seg_img_score = self.dataset[index]
            img = read_image(img_path).resize(img_size)
            img_2 = read_image(img_path).resize(img_size)
            seg_img = [Image.open(img_path) for img_path in seg_img_name]
            seg_img_score = [i['score'] for i in seg_img_score]
            part_num = len(seg_img)
            if part_num >= self.max_part_num:
                seg_img = seg_img[:self.max_part_num]
                seg_img_score = seg_img_score[:self.max_part_num]
                attribute = attribute[:self.max_part_num]
                part_num = self.max_part_num
            else:
                attribute = attribute + [attribute[0] for i in range(self.max_part_num-part_num)]

            merged_mask = Image.fromarray(merge_mask(seg_img,seg_img_score)).resize(img_size)
            merged_mask_2 = Image.fromarray(merge_mask(seg_img,seg_img_score)).resize(img_size)

            #img,merged_mask = random_apply(img,merged_mask,random_transforms)
            for func in function_list:
                img,merged_mask = func(img,merged_mask)
                img_2,merged_mask_2 = func(img_2,merged_mask_2)

            if not isinstance(img,Image.Image):
                merged_mask = (merged_mask*255).long().numpy()
            else:
                img = tf.to_tensor(img)

            if not isinstance(img_2,Image.Image):
                merged_mask_2 = (merged_mask_2*255).long().numpy()
            else:
                img_2 = tf.to_tensor(img_2)
            
            img_norm = F.normalize(img, mean, std, False)
            img_2_norm = F.normalize(img_2, mean, std, False)
            
            label_list = generate_token_class(np.array(merged_mask))
            label = torch.LongTensor(label_list)

            label_list_2 = generate_token_class(np.array(merged_mask_2))
            label_2 = torch.LongTensor(label_list_2)

            caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

            mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())
            #attribute_tokens_wospecial = [self.tokenizer.encode(attr) for attr in attribute]
            attr_index = []
            for attr_id in [self.tokenizer.encode(attr) for attr in attribute]:
                attr_index.append(find_subsequence_index_with_error(caption_tokens,attr_id)[0])

            attr_index = torch.LongTensor(attr_index)

            mlm_part_ids,mlm_part_labels = self._build_part_random_masked_tokens_and_labels(
                caption_tokens.cpu().numpy(),
                attr_index,
                part_mask_prob=self.part_mask_prob,
                mask_prob=self.mask_prob
            )

            ret = {
                'pids': pid,
                'image_ids': image_id,
                'images': img_norm,
                'images_2': img_2_norm,
                'caption_ids': caption_tokens,
                'mlm_ids': mlm_tokens,
                'mlm_labels': mlm_labels,
                'label':label,
                'label_2':label_2,
                'part_num':part_num,
                "attr_index":attr_index,
                "mlm_part_ids":mlm_part_ids,
                "mlm_part_labels":mlm_part_labels
            }

            if self.using_mim:
                ret['images_origin'] = img
            if self.return_attr_tokens:
                attribute_tokens = torch.stack([tokenize(atr, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate) for atr in attribute],dim=0)
                ret['attribute'] = attribute_tokens

            return ret
        else:
            pid, image_id, img_path, caption = self.dataset[index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            
            caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

            mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

            ret = {
                'pids': pid,
                'image_ids': image_id,
                'images': img,
                'caption_ids': caption_tokens,
                'mlm_ids': mlm_tokens,
                'mlm_labels': mlm_labels
            }

            return ret

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)
    
    def _build_part_random_masked_tokens_and_labels(self, tokens, part_index, part_mask_prob=0.5, mask_prob=0.15):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        def isin(idx,part_index):
            for rng in part_index:
                if idx in rng:
                    return True
            return False
        
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                temp_prob = 0.0
                if isin(i,part_index):
                    temp_prob = part_mask_prob
                    pass
                else:
                    temp_prob = mask_prob
                    pass
                if prob < temp_prob:
                    prob /= temp_prob

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)