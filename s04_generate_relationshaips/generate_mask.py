import os
import torch
import json
import argparse

from tqdm import tqdm
from PIL import Image

import numpy as np
import torch.nn.functional as F

from transformers import SamImageProcessor
from transformers import BertModel
from transformers import AutoTokenizer

from models.PersonSAM import PersonSAM
from datasets.build import __factory

class training_options():
    def __init__(self):
        self._par = argparse.ArgumentParser(description='options for datasets')
        self._par.add_argument('--debug',default=False, type=bool)

        self._par.add_argument('--batch_size',default=2, type=int)
        self._par.add_argument('--num_epochs',default=20, type=int)
        self._par.add_argument('--sentence_max_len',default=16, type=int)

        self._par.add_argument('--base_lr',default=1e-4, type=float)
        self._par.add_argument('--weight_decay',default=0.01, type=float)

        self._par.add_argument('--bias_lr_factor',default=1.0, type=float)
        self._par.add_argument('--weight_decay_bias',default=0.0, type=float)
        self._par.add_argument('--optimizer',default="AdamW", type=str)
        self._par.add_argument('--sgd_momentum',default=0.9, type=float)
        self._par.add_argument('--adam_alpha',default=0.9, type=float)
        self._par.add_argument('--adam_beta',default=0.999, type=float)

        self._par.add_argument('--steps',default=(500,), type=list)
        self._par.add_argument('--gamma',default=0.1, type=float)
        self._par.add_argument('--warmup_factor',default=0.1, type=float)
        self._par.add_argument('--warmup_epochs',default=1, type=int)
        self._par.add_argument('--warmup_method',default="linear", type=str)
        self._par.add_argument('--lrscheduler',default="step", type=str)
        self._par.add_argument('--target_lr',default=0.0002, type=float)
        self._par.add_argument('--power',default=0.9, type=float)

        self._par.add_argument('--loss_type',default="DiceCELoss", type=str)
        self._par.add_argument('--sigmoid',default=True, type=bool)
        self._par.add_argument('--squared_pred',default=True, type=bool)
        self._par.add_argument('--data_aug',default=False, type=bool)
        self._par.add_argument('--reduction',default="mean", type=str)

        self._par.add_argument('--data_path',default="./humanparsing", type=str)
        self._par.add_argument('--sam_path',default="facebook/sam-vit-base", type=str)
        self._par.add_argument('--language_model_path',default="bert-base-uncased", type=str)


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


class dataset_options():
    def __init__(self):
        self._par = argparse.ArgumentParser(description='options for Deep Cross Modal')
        self._par.add_argument('--dataset', type=str)
        self._par.add_argument('--wordtype', type=str)
        self._par.add_argument('--pkl_root', type=str)
        self._par.add_argument('--class_num', type=int)
        self._par.add_argument('--vocab_size', type=int)
        self._par.add_argument('--dataroot', type=str)
        self._par.add_argument('--mode', type=str)
        self._par.add_argument('--attr_num',default=6, type=int)
        self._par.add_argument('--attr_len',default=16, type=int)
        self._par.add_argument('--batch_size', type=int)
        self._par.add_argument('--sam_processor_path', type=str)
        self._par.add_argument('--caption_length_max', type=int)


class options():
    def __init__(self):
        self._par = argparse.ArgumentParser(description='options for datasets')
        self._par.add_argument('--step',default=1000, type=int)
        self._par.add_argument('--begin',default=0, type=int)
        self._par.add_argument('--dataset',default="CUHK-PEDES", type=str)
        self._par.add_argument('--data_path',default="Datasets/", type=str)
        self._par.add_argument('--bert_path',default="bert-base-uncased", type=str)
        self._par.add_argument('--sam_path',default="F:\preTrainedModels\sam-vit-base", type=str)
        self._par.add_argument('--trained_sam',default=r"F:\atr-sam\atrsamv2\best.pth", type=str)

# CUDA_VISIBLE_DEVICES=7 python generate_mask.py --data_path ../sam-for-tbpr/ --bert_path ./bert-base-uncased/ --sam_path ./sam-vit-base-tbpr/ --trained_sam ../../best.pth --step 1000000

if __name__ == "__main__":
    opt = options()._par.parse_args()

    ############ load sam model
    args = training_options()._par.parse_args([
        "--batch_size","1",
        "--sam_path",opt.sam_path,
        "--language_model_path",opt.bert_path
    ])

    tokenizer = AutoTokenizer.from_pretrained(args.language_model_path)
    processor = SamImageProcessor.from_pretrained(args.sam_path)

    sam_model = PersonSAM.from_pretrained(args.sam_path)
    text_prompt_encoder = BertModel.from_pretrained(args.language_model_path)
    sam_model.set_pretrained_text_encoder(text_prompt_encoder)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam_model.load_state_dict(torch.load(opt.trained_sam))
    sam_model = sam_model.eval()

    # set frozen parameters
    for name, param in sam_model.named_parameters():
        param.requires_grad_(False)

    sam_model = sam_model.to(device)

    ######## load dataset
    dataset = __factory[opt.dataset](root=opt.data_path)
    sentence_dict_name = {
        "CUHK-PEDES":"CUHK-PEDES_data_final_68126.json",
        "ICFG-PEDES":"ICFG-PEDES_data_final_34674.json",
        "RSTPReid":"RSTPReid_data_final_37010.json"
    }

    with open(os.path.join(opt.data_path,opt.dataset,sentence_dict_name[opt.dataset]),"r+") as f:
        data = json.load(f)

    score_dict = {}
    # path = dataset.dataset_dir+"\\segs\\"
    path = os.path.join(dataset.dataset_dir,"segs")
    #for idx in tqdm(range(10000,cuhk_train_dataloader.dataset.__len__())):
    for item_data in tqdm(data):
        idx = item_data['idx']
        item = dataset.train[idx]
        image = Image.open(item[-2])
   
        attribute = item_data['attribute']
        #There are times when the data will appear to be [['sentence']]
        if attribute == []:
            continue
        if isinstance(attribute[0],list):
            attribute = attribute[0]
        
        flag = False
        for i,atr in enumerate(attribute):
            predict_original_image,score,predict_mask = inference_on_tbpr(sam_model,args,image,atr,processor,tokenizer,device)
            #plot_inference_image(score,image,predict_original_image,atr)
            name = dataset.train[idx][-2][len(dataset.dataset_dir+"imgs/"):-len(".png")]
            name = "_".join([str(idx)]+name.split("/")+[str(i)])
            seg_mask = Image.fromarray(predict_mask)
            seg_mask.save(os.path.join(path,name+".png"))
            #seg_mask.save(path+name+".png")
            score_dict[name] = {"score":score,"text":atr}
            #print({"score":score,"text":atr})
        #break
    with open(opt.dataset+"_score_dict_{}.json".format(idx),"w+") as f:
        json.dump(score_dict,f)