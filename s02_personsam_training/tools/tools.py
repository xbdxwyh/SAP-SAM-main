
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import torch

import torch.nn.functional as F

from . import dataAugment

def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox

def get_transform_aug(data_type="train"):
    aug_list = [
        dataAugment.ImageMaskRandomHFlip(p=0.5),# OK
        dataAugment.ImageMaskRandomGrayscale(p=0.5), # OK
        dataAugment.ImageMaskColorJitter(brightness=.15, hue=.1), # OK
        dataAugment.ImageMaskGaussianBlur(kernel_size=(3, 5), sigma=(0.02, 1.5)),  # OK
        dataAugment.ImageMaskRandomInvert(), # OK
        dataAugment.ImageMaskRandomPosterize(bits=3), # OK
        dataAugment.ImageMaskRandomAdjustSharpness(sharpness_factor=4),# OK
        dataAugment.ImageMaskRandomAutocontrast(), # OK
        dataAugment.ImageMaskAugMix(severity=3),# OK, 版本原因
        dataAugment.ImageMaskRandomRotation(25),# OK
        dataAugment.ImageMaskRandomAffine(degrees=(5, 15),translate=(0,0),shear=10), # OK
        dataAugment.ImageMaskRandomEqualize(), # OK
    ]
    return dataAugment.ImageMaskRandomApply(aug_list,p=0.3)


class SAMATRDataset(Dataset):
    def __init__(self,
            data_path, 
            data = "train", 
            sentence_length=16, 
            processor=None, 
            tokenizer=None,
            data_type="train",
            transform_aug = None
        ):
        self.data_path = data_path
        self.images = os.listdir(data_path+"//JPEGImages")
        self.ground_truth = os.listdir(data_path+"//SegmentationClassAug")
        if transform_aug is None and data_type=="train":
            self.transform_aug = get_transform_aug(data_type)
        else:
            self.transform_aug = transform_aug
            
        print(self.transform_aug)
        self.data_type = data_type

        with open(data_path+"//atr_label.txt","r") as f:
            lines = f.readlines()
        
        with open (data_path+"//atr_item_descriptions_{}.json".format(data),'r') as f:
            descriptions = json.load(f)

        captions = []
        for line in lines:
            line = line.split("\n")[0]
            captions.append(line.split(" ")[0])
            #labels[idx] = caption
        
        self.semantic_combine = {
            '1':[1],
            '2':[2],
            '3':[3],
            '4':[4],
            '5':[5],
            '6':[6],
            '7':[7],
            '9':[9,10],
            '16':[16]
        }

        self.labels = {k:v for k,v in zip(range(len(captions)),captions)}
        self.descriptions = descriptions
        self.sentence_length = sentence_length
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        # get item data and image 
        item = self.descriptions[idx]
        image_name = item['name']
        ground_truth_name = image_name.split(".")[0]+".png"

        image = Image.open(self.data_path + "//JPEGImages//" + image_name)
        ground_truth_mask = Image.open(self.data_path + "//SegmentationClassAug//" + ground_truth_name)
        if self.data_type == "train":
            image,ground_truth_mask = self.transform_aug(image,ground_truth_mask)
            pass

        ground_truth_mask = np.array(
            ground_truth_mask.resize((
                self.processor.size['longest_edge']//4,
                self.processor.size['longest_edge']//4
            )
        ))

        semantic_id_list = self.semantic_combine[item['idx']]

        semantic_mask = np.zeros_like(ground_truth_mask)
        for i in semantic_id_list:
            semantic_mask = semantic_mask + (ground_truth_mask==i).astype(np.uint8)
        
        label = torch.LongTensor([semantic_id_list[0]])
        
        # prepare image and prompt for the model
        image_inputs = self.processor(image, return_tensors="pt")

        # remove batch dimension which the processor adds by default
        image_inputs = {k:v.squeeze(0) for k,v in image_inputs.items()}

        # add ground truth segmentation
        image_inputs["ground_truth_mask"] = torch.Tensor(semantic_mask)

        # add processed text prompt
        text_inputs = self.tokenizer(
            item['text'],
            padding="max_length",
            max_length=self.sentence_length,
            truncation=True,
            return_tensors="pt"
        )

        inputs = {}
        for k,v in image_inputs.items():
            inputs[k] = v
        for k,v in text_inputs.items():
            inputs[k] = v.squeeze(0)
        
        inputs['semantic_label'] = label

        return inputs


#def get_detail(dataset,idx,semantic=[4],pad=0,crop_bbox=False):
def get_detail(data_path,images,ground_truth,idx,semantic=[4],pad=0,crop_bbox=False):
    # get image
    image = Image.open(data_path + "//JPEGImages//" + images[idx])
    # get labels
    ground_truth_mask = Image.open(data_path + "//SegmentationClassAug//" + ground_truth[idx])
    ground_truth_mask = np.array(ground_truth_mask)
    # create a image copy
    original_image = np.array(image)
    original_image_mask = np.zeros_like(ground_truth_mask)
    # plot using different color
    for i in semantic:
        original_image_mask = original_image_mask + (ground_truth_mask==i).astype(np.uint8)
    
    mask_2d = original_image_mask
    if len(original_image.shape) > 2:
        original_image_mask = original_image_mask[:,:,None].repeat(3,axis=-1)
        segmentation_image = np.where(original_image_mask!=1, np.full_like(original_image, [pad,pad,pad]  ), original_image)
    else:
        original_image_mask = original_image_mask
        segmentation_image = np.where(original_image_mask!=1, np.full_like(original_image, [pad]  ), original_image)
    
    bbox = get_bounding_box(mask_2d)
    
    if crop_bbox:
        return Image.fromarray(segmentation_image).crop(bbox)
    else:
        return Image.fromarray(segmentation_image),bbox


def plot_image_and_semantic(data_path,images,ground_truth,idx,semantic_list = None):
    # define color list
    color_list = [
        [30,144,255], # aorta
        [0,255,0],    # gallbladder
        [255,0,0],    # left kidney
        [0,255,255],  # right kidney
        [255,0,255],  # liver
        [255,255,0],  # pancreas
        [128,0,255],  # spleen
        [255,128,0],  # stomach
        [51,204,153],
        [255,153,153],
        [255,255,153],
        [204,255,153],
        [204,204,51],
        [204,153,204],
        [153,153,153],
        [153,204,255],
        [51,51,255],
    ]
    # get image
    image = Image.open(data_path + "//JPEGImages//" + images[idx])
    # get labels
    ground_truth_mask = Image.open(data_path + "//SegmentationClassAug//" + ground_truth[idx])
    ground_truth_mask = np.array(ground_truth_mask)
    # create a image copy
    original_image = np.array(image)
    # set semantic to prepare, default to plot all segmentation
    if semantic_list is None:
        semantic_list = np.unique(ground_truth_mask)
    # plot using different color
    for i in semantic_list:
        if len(original_image.shape) > 2:
            test_mask = ground_truth_mask[:,:,None].repeat(3,axis=-1)
            original_image = np.where(test_mask==i, np.full_like(original_image, color_list[i]), original_image)
        else:
            test_mask = ground_truth_mask
            original_image = np.where(test_mask==i, np.full_like(original_image, [color_list[i][0]]  ), original_image)

    original_image = Image.fromarray(original_image)
    fig, axes = plt.subplots(1, 2, figsize=(15,15))
    axes[0].imshow(image)
    axes[0].title.set_text("original_image")
    axes[0].axis("off")
    
    axes[1].imshow(original_image)
    axes[1].title.set_text("semantic segmentation")
    axes[1].axis("off")
    
    plt.show()



def inference_on_tbpr(sam_model,args,tbpr_image,text_prompt,processor,tokenizer,device,color = [30,144,255]):
    image_inputs = processor(tbpr_image, return_tensors="pt")
    image_inputs = {k:v.squeeze(0) for k,v in image_inputs.items()}
    text_inputs = tokenizer(text_prompt,padding="max_length",max_length=args.sentence_max_len,truncation=True ,return_tensors="pt")
    inputs = {}
    for k,v in image_inputs.items():
        inputs[k] = v
    for k,v in text_inputs.items():
        inputs[k] = v.squeeze(0)
    
    outputs = sam_model(
        pixel_values=inputs['pixel_values'].unsqueeze(0).to(device),
        input_ids = inputs['input_ids'].unsqueeze(0).to(device),
        token_type_ids = inputs['token_type_ids'].unsqueeze(0).to(device),
        attention_mask = inputs['attention_mask'].unsqueeze(0).to(device),
        multimask_output=False
    )
    
    score = outputs.iou_scores.detach().cpu().numpy().tolist()[0][0][0]
    # post process predicted results
    image_shape = tuple(inputs['original_sizes'].numpy().tolist())

    predict_mask = (F.interpolate(outputs.pred_masks.cpu()[0],image_shape, mode="bilinear").numpy()>0).astype(np.uint8)[0][0]
    
    # plot predict results on origin image
    predict_original_image = np.array(tbpr_image)
    # predict_mask
    if len(predict_original_image.shape) > 2:
        test_mask = predict_mask[:,:,None].repeat(3,axis=-1)
        predict_original_image = np.where(test_mask!=0, np.full_like(predict_original_image, color), predict_original_image)
    else:
        test_mask = predict_mask
        predict_original_image = np.where(test_mask!=0, np.full_like(predict_original_image, [color[0]]  ), predict_original_image)

    predict_original_image = Image.fromarray(predict_original_image)
    
    return predict_original_image,score,predict_mask



def plot_inference_image(score,image,predict_original_image,text):
    fontsize = 18
    fig, axes = plt.subplots(1, 2, figsize=(8.5,6))
    
    axes[0].imshow(image)
    axes[0].set_title("Origin Image",fontsize=fontsize)
    axes[0].axis("off")

    axes[1].imshow(predict_original_image)
    axes[1].set_title("Predictions",fontsize=fontsize)
    axes[1].axis("off")
    
    plt.suptitle("Text Description: {}, Prediction Score: {:.2f}".format(text,score*100),fontsize=fontsize)


def add_mask2image(image,mask,color=[[30,144,255],[145,16,255],[207,144,44],[12,144,41]]):
    predict_original_image = np.array(image)
    predict_mask = np.array(mask)
    max_semantic = predict_mask.max()
    if len(color)<max_semantic:
        color = color+[color[0]] * (max_semantic-len(color)+1)

    for i in range(1,max_semantic+1):
        # predict_mask
        if len(predict_original_image.shape) > 2:
            test_mask = predict_mask[:,:,None].repeat(3,axis=-1)
            predict_original_image = np.where(test_mask==i, np.full_like(predict_original_image, color[i]), predict_original_image)
        else:
            test_mask = predict_mask
            predict_original_image = np.where(test_mask==i, np.full_like(predict_original_image, [color[i][0]]  ), predict_original_image)

    predict_original_image = Image.fromarray(predict_original_image)
    return predict_original_image
    pass


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

def find_subsequence_index(sentence, part):
    """
    Args:
    sentence: 长序列。
    part: 子序列。

    Returns:
    子序列在长序列中的位置列表。
    """
    index = []
    for i in range(sentence.shape[-1]):
        if sentence[i] == part[0]:
            match = True
            for j in range(1, len(part)):
                if sentence[i + j] != part[j]:
                    match = False
                    break
            if match:
                index.append((i, i + len(part) - 1))
    return index

def find_phrase_index(sentence_id,phrase_id):
    sentence_len,phrase_len = sentence_id.shape[-1],len(phrase_id)
    begin,end = 0,0
    w_index = 0
    begin_id = phrase_id[w_index]
    w_p = phrase_id[w_index]
    for i in range(sentence_len):
        w_s = sentence_id[i]
        if w_s == begin_id:
            begin = i
        if w_s == w_p:
            w_index += 1
            #print(w_index)
            end = i
            if w_index >= phrase_len:
                #return begin,end
                break
            w_p = phrase_id[w_index]
    return begin,end

# inputs = processor(images=image,text="Question: What kind of hat is this? Answer:", return_tensors="pt").to(device, torch.float16) # 用于1

# inputs = processor(images=image,text="Question: What's her hair style? Answer:", return_tensors="pt").to(device, torch.float16)
# inputs = processor(images=image,text="Question: What kind of hair she has? Answer:", return_tensors="pt").to(device, torch.float16)
# inputs = processor(images=image,text="Question: What kind of hair is this? Answer:", return_tensors="pt").to(device, torch.float16)
# inputs = processor(images=image,text="Question: What kind of hair is she? Answer:", return_tensors="pt").to(device, torch.float16) 
# inputs = processor(images=image,text="Question: What kind of hair is she? Answer: It is", return_tensors="pt").to(device, torch.float16)# 用于2,11

# inputs = processor(images=image,text="Question: What kind of glasses is this? Answer:", return_tensors="pt").to(device, torch.float16) # 用于3,11

# inputs = processor(images=image,text="Question: What kind of clothes is this? Answer:", return_tensors="pt").to(device, torch.float16) # 用于4

# inputs = processor(images=image,text="Question: What kind of skirt is this? Answer:", return_tensors="pt").to(device, torch.float16) # 用于5

# inputs = processor(images=image,text="Question: What kind of pants is this? Answer: It is", return_tensors="pt").to(device, torch.float16) # 用于6

# inputs = processor(images=image,text="Question: What kind of dress is this? Answer:", return_tensors="pt").to(device, torch.float16) # 用于7

# inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
# inputs = processor(images=image,text="Question: What object is in this picture? Answer:", return_tensors="pt").to(device, torch.float16)
# inputs = processor(images=image,text="Question: What's in this picture? Answer:", return_tensors="pt").to(device, torch.float16)
# inputs = processor(images=image,text="Question: What kind of shoes are these? Answer:", return_tensors="pt").to(device, torch.float16) #
# inputs = processor(images=image,text="Question: What shoes is she wearing? Answer:", return_tensors="pt").to(device, torch.float16)# 用于9,10,12,13

# inputs = processor(images=image,text="Question: What kind of bag is this? Answer:", return_tensors="pt").to(device, torch.float16) # 用于16
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