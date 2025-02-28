
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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


class SAMLVMHPv1Dataset(Dataset):
    def __init__(self, data_path, sentence_length=16, processor=None, tokenizer=None):
        self.data_path = data_path
        self.images = os.listdir(data_path+"//images")
        self.ground_truth = os.listdir(data_path+"//annotations")
        with open(data_path+"//atr_label.txt","r") as f:
            lines = f.readlines()
        
        captions = []
        for line in lines:
            line = line.split("\n")[0]
            captions.append(line.split(" ")[0])
            #labels[idx] = caption

        self.labels = {k:v for k,v in zip(range(len(captions)),captions)}
        self.sentence_length = sentence_length
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        #item = self.dataset[idx]
        image = Image.open(self.data_path + "//JPEGImages//" + self.images[idx])
        ground_truth_mask = np.array(Image.open(self.data_path + "//SegmentationClassAug//" + self.ground_truth[idx]).resize((256,256)))
        
        # prepare image and prompt for the model
        inputs = self.processor(image, input_points=[[[10,10]]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

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

