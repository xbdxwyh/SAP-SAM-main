import torch
import argparse
import monai
import random
import datetime

import time
import numpy as np

from tqdm import tqdm
from statistics import mean

from tools.solver import make_lr_scheduler, make_optimizer
from torch.utils.data import DataLoader

from transformers import SamImageProcessor
from transformers import BertModel
from transformers import AutoTokenizer

from tools.logger import setup_logger
from models.PersonSAM import PersonSAM
from tools.tools import SAMATRDataset

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'Total Params : {:.2f} M'.format(total_num/1000000),
        'Trainable Params : {:.2f} M'.format(trainable_num/1000000), 
        "Trainable/Total : {:.2f} %".format(trainable_num/total_num*100)
    }


def binary_iou(s, g):
    #assert (len(s.shape)  len(g.shape))
    # 两者相乘值为1的部分为交集
    intersecion = np.multiply(s, g)
    # 两者相加，值大于0的部分为交集
    union = np.asarray(s + g > 0, np.float32)
    
    iou = intersecion.sum() / (union.sum() + 1e-10)
    return iou

def evaluation(sam_model,args,dataloader,device,semantic_label = [1,2,3,4,5,6,7,9,16]):
    iou_list = []
    iou_list_with_label = [[] for i in range(17)]
    for batch in tqdm(dataloader):
        with torch.no_grad():
            outputs = sam_model(
                pixel_values=batch['pixel_values'].to(device),
                input_ids = batch['input_ids'].to(device),
                token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch.keys() else None,
                attention_mask = batch['attention_mask'].to(device),
                multimask_output=False
            )
        # compute IOU  
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)

        # masks = processor.post_process_masks(
        #     outputs.pred_masks.cpu(), original_sizes, original_sizes
        # )
        masks = outputs.pred_masks.cpu()>0
        num_samples = masks.shape[0]

        for i in range(num_samples):
            s = masks[i][0][0].numpy().astype(np.uint8)
            g = ground_truth_masks[i].cpu().numpy()
            iou = binary_iou(s,g)
            iou_list.append(iou)   
            iou_list_with_label[batch['semantic_label'][i].numpy().tolist()[0]].append(iou)
        if args.debug:
            break
    
    return iou_list,iou_list_with_label

def train(sam_model,args,logger,processor,device,optimizer,scheduler,seg_loss,train_dataloader,dev_dataloader,test_dataloader,log_period=100):
    semantic_label = [1,2,3,4,5,6,7,9,16]
    best_iou = 0.0
    meters = {
        "loss": AverageMeter(),
    }
    for epoch in range(args.num_epochs):
        #epoch_losses = []
        sam_model.train()
        logger.info("Training Epoch: {}/{}".format(epoch,args.num_epochs))
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        for step,batch in tqdm(enumerate(train_dataloader)):
            # forward pass
            outputs = sam_model(
                pixel_values=batch['pixel_values'].to(device),
                input_ids = batch['input_ids'].to(device),
                token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch.keys() else None,
                attention_mask = batch['attention_mask'].to(device),
                multimask_output=False
            )

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # optimize
            optimizer.step()
            meters['loss'].update(loss.item(), args.batch_size)
            #epoch_losses.append(loss.item())
            if (step + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{step + 1}/{len(train_dataloader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
            #epoch_losses.append(loss.item())
            #logger.info("Epoch: {}/{}, Step: {}/{}, Loss: {}".format(epoch,args.num_epochs,step,train_dataloader.dataset.__len__()//args.batch_size,loss.item()))
            if args.debug:
                break
        
        sam_model.eval()
        logger.info("Evaluation Epoch: {}/{}".format(epoch,args.num_epochs))
        
        iou_list,iou_list_with_label = evaluation(sam_model,args,dev_dataloader,device,semantic_label)
        
        iou_mean = mean(iou_list)
        if iou_mean >= best_iou:
            best_iou = iou_mean
            torch.save(sam_model.state_dict(),"best.pth")
        
        #print(f'EPOCH: {epoch}')
        #print(f'Mean loss: {mean(epoch_losses)}')
        logger.info("Training Epoch: {}/{}, Mean IoU: {}, best_iou:{}".format(epoch+1,args.num_epochs,mean(iou_list),best_iou))
        logger.info("Max IoU: {}, Min IoU: {}".format(max(iou_list),min(iou_list)))
        logger.info(" ".join(["Semantic Class: {}, Mean IoU: {}, Max IoU: {}, Min IoU: {}, Sample Num: {};\n".format(i,mean(iou_list_with_label[i]),max(iou_list_with_label[i]),min(iou_list_with_label[i]),len(iou_list_with_label[i])) for i in semantic_label]))
        scheduler.step()
        if args.debug:
            break
    
    iou_list,iou_list_with_label = evaluation(sam_model,args,test_dataloader,device,semantic_label)
    logger.info("evaluation On test dataset!")
    logger.info("Mean IoU: {}".format(mean(iou_list)))
    logger.info("Max IoU: {}, Min IoU: {}".format(max(iou_list),min(iou_list)))
    logger.info(" ".join(["Semantic Class: {}, Mean IoU: {}, Max IoU: {}, Min IoU: {}, Sample Num: {};\n".format(i,mean(iou_list_with_label[i]),max(iou_list_with_label[i]),min(iou_list_with_label[i]),len(iou_list_with_label[i])) for i in semantic_label]))



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


def main(args):
    ts = datetime.datetime.now().timestamp()
    logger = setup_logger("SAMtuningOnATR","./",0,"log_{}.txt".format(str(ts)))
    logger.info(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.language_model_path)
    processor = SamImageProcessor.from_pretrained(args.sam_path)

    if args.data_aug:
        train_set = SAMATRDataset(args.data_path,data="train",data_type="train",processor=processor,tokenizer=tokenizer)
    else:
        train_set = SAMATRDataset(args.data_path,data="train",data_type="dev",processor=processor,tokenizer=tokenizer)
    dev_set = SAMATRDataset(args.data_path,data="dev",data_type="dev",processor=processor,tokenizer=tokenizer)
    test_set = SAMATRDataset(args.data_path,data="test",data_type="test",processor=processor,tokenizer=tokenizer)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,num_workers=12)
    dev_dataloader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False,num_workers=12)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,num_workers=12)

    # prepare model
    sam_model = PersonSAM.from_pretrained(args.sam_path)
    text_prompt_encoder = BertModel.from_pretrained(args.language_model_path)
    sam_model.set_pretrained_text_encoder(text_prompt_encoder)

    # set frozen parameters
    for name, param in sam_model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder") \
            or name.startswith("text_prompt_encoder.embeddings") or name.startswith("text_prompt_encoder.encoder"):
            param.requires_grad_(False)
    
    sam_model = sam_model.to(device)
    logger.info(get_parameter_number(sam_model))

    # set trainable optimizer
    trainable_params = []
    for pname, p in sam_model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder") \
        or name.startswith("text_prompt_encoder.embeddings") or name.startswith("text_prompt_encoder.encoder"):
            pass
        else:
            trainable_params += [p]
    
    # Note: Hyperparameter tuning could improve performance here
    #optimizer = Adam(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = make_optimizer(args, sam_model)
    scheduler = make_lr_scheduler(args, optimizer)

    if args.loss_type == "DiceCELoss":
        seg_loss = monai.losses.DiceCELoss(
            sigmoid=args.sigmoid, 
            squared_pred=args.squared_pred, 
            reduction=args.reduction
        )
    else:
        raise NotImplementedError("Unknown Loss Name!")

    train(
        sam_model,
        args,
        logger,
        processor,
        device,
        optimizer,
        scheduler,
        seg_loss,
        train_dataloader,
        dev_dataloader,
        test_dataloader
    )
    pass

if __name__ == "__main__":
    set_random_seed()
    args = training_options()._par.parse_args()
    main(args)
