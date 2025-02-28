from PIL import Image
import numpy as np
import torch
import json
import argparse
import os

from tqdm import tqdm

from transformers import Blip2Processor, Blip2ForConditionalGeneration

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
    

class options():
    def __init__(self):
        self._par = argparse.ArgumentParser(description='options for datasets')
        self._par.add_argument('--step',default=1000000, type=int)
        self._par.add_argument('--begin',default=0, type=int)
        self._par.add_argument('--data_path',default="./humanparsing", type=str)
        self._par.add_argument('--blip_path',default="Salesforce/blip2-opt-2.7b", type=str)


if __name__ == "__main__":
    opt = options()._par.parse_args()
    print("Begin processing!")
    data_path = opt.data_path
    blip_path = opt.blip_path
    
    images = os.listdir(data_path+"//JPEGImages")
    ground_truth = os.listdir(data_path+"//SegmentationClassAug")

    semantic_combine = [
        [1],
        [2,11],
        [3,11],
        [4],
        [5],
        [6],
        [7],
        [9,10,12,13],
        [16]
    ]

    prompt_list = [
        "Question: What kind of hat is this? Answer: It is",
        "Question: What kind of hair is she? Answer: She has",
        "Question: What kind of glasses is this? Answer: It is",
        "Question: What kind of clothes is this? Answer: It is",
        "Question: What kind of skirt is this? Answer: It is",
        "Question: What kind of pants is this? Answer: It is",
        "Question: What kind of dress is this? Answer: It is",
        "Question: What shoes is she wearing? Answer: She is wearing",
        "Question: What kind of bag is this? Answer: It is",
    ]

    blip_processor = Blip2Processor.from_pretrained(blip_path)
    model = Blip2ForConditionalGeneration.from_pretrained(blip_path, torch_dtype=torch.float16)
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    try:
        description_json = []
        end = min(opt.begin+opt.step,len(images))
        for i in tqdm(range(opt.begin,end)):
            try:
                item_dict = {}
                unique_mask = np.unique(np.array(Image.open(data_path + "//SegmentationClassAug//" + ground_truth[i])))

                for semantic,prompt in zip(semantic_combine,prompt_list):
                    if semantic[0] not in unique_mask:
                        #print("id:",i,",semantic:",semantic,",continue")
                        continue
                    image = get_detail(data_path,images,ground_truth,i,semantic=semantic,pad=255,crop_bbox=True)
                    #image = Image.open("test.jpg")
                    inputs = blip_processor(images=image,text=prompt, return_tensors="pt").to(device, torch.float16)
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs)
                    
                    generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    item_dict[semantic[0]] = generated_text
            except ValueError:
                with open("err_list.txt", 'w+') as f:
                    f.write("{}\n".format(i))
                continue
            except:
                break
            else:
                #item_dict['name'] = dataset.data_path[i]
                description_json.append({"description":item_dict,"name":images[i],"id":i})
    # except:
    #     with open("err_list.txt", 'w') as  f:
    #         f.write("{}\n".format(opt.begin))
    #         # if i > 250:
    #         #     break
    finally:
        with open("descriptions//description_json_{}_{}.json".format(opt.begin,i), 'w') as  f:
            json.dump(description_json, f)
    
        print("End processing!")
