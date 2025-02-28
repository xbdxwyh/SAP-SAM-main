from . import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

import monai
from monai.losses.dice import one_hot

import copy
import torch.nn.functional as F

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        self.eps=1e-2

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mim' in args.loss_names or 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(
                self.embed_dim,
                self.embed_dim // 64,
                batch_first=True
            )
            self.cross_modal_transformer = Transformer(
                width=self.embed_dim,
                layers=args.cmt_depth,
                heads=self.embed_dim // 64
            )
            
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)
        
        # if "mim_part" in args.loss_names:
        #     self.mim_loss = nn.MSELoss()
        #     self.mim_part_mask_token = nn.Parameter(torch.randn(1,self.embed_dim))
        #     self.mim_decoder = nn.Linear(self.embed_dim,self.embed_dim)
        #     pass
        if "mim_part" in args.loss_names:
            self.mim_loss = nn.MSELoss()
            #self.mim_part_mask_token = nn.Parameter(torch.randn(1,self.embed_dim))
            patch_dim = self.base_model.visual.ln_pre.weight.shape[0]
            self.mim_part_mask_token = nn.Parameter(torch.randn(1,patch_dim))
            self.mim_decoder = nn.Linear(self.embed_dim,patch_dim)

        if 'matching' in args.loss_names:
            self.classifier_matching = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, 2))]))
            #torch.nn.Linear(in_features=self.embed_dim,out_features=2,dtype=torch.float16)
            self.cross_attn_matching = nn.MultiheadAttention(
                self.embed_dim,
                self.embed_dim // 64,
                batch_first=True
            )
            # self.img_cls_token = nn.Parameter(torch.randn(1,1,self.embed_dim))
            # self.txt_cls_token = nn.Parameter(torch.randn(1,1,self.embed_dim))
            # pass
            fc_std = (2 * self.embed_dim)**-0.5
            scale = self.embed_dim**-0.5
            proj_std = scale * ((2 * args.cmt_depth)**-0.5)
            attn_std = scale
            # nn.init.normal_(self.classifier_seg.weight, std=fc_std)
            nn.init.normal_(self.classifier_matching.dense.weight, std=fc_std)
            nn.init.normal_(self.classifier_matching.fc.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn_matching.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn_matching.out_proj.weight, std=proj_std)

        if 'mlm' in args.loss_names:
            scale = self.cross_modal_transformer.width**-0.5
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    
    def cross_former_image(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_i(q),
                self.ln_pre_t(k),
                self.ln_pre_t(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    
    def query_cross_former(self, q, k, v):
        x = self.query_cross_attn(
                self.query_ln_pre_t(q),
                self.query_ln_pre_i(k),
                self.query_ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.query_cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.query_ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        if "augtext" in self.current_task:
            caption_ids = batch['mlm_ids']
        else:
            caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        # i_feats_norm = F.normalize(i_feats)
        # t_feats_norm = F.normalize(t_feats)

        pid = batch['pids']
        pid = pid.view(-1, 1)
        pid_all = pid.view(1, -1)
        pos_idx = torch.eq(pid, pid_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        sim_targets = sim_targets.to(image_feats.device)

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            with torch.no_grad():
                image_feats_s, text_feats_s = self.base_model(images, caption_ids)
                i_feats_s = image_feats_s.detach()[:, 0, :].float()
                # i_feats = image_feats.float() # for CLIP ResNet visual model
                t_feats_s = text_feats_s.detach()[torch.arange(text_feats_s.shape[0]), caption_ids.argmax(dim=-1)].float()
                # i_feats_s_norm = F.normalize(i_feats_s)
                # t_feats_s_norm = F.normalize(t_feats_s)
            ret.update({'itc_loss':objectives.compute_n_itc(i_feats, t_feats, i_feats_s, t_feats_s, batch['pids'], logit_scale)})
        
        if 'sdm' in self.current_task:
            if "ss" in self.current_task:
                images_2 = batch['images_2']
                image_feats_2 = self.base_model.encode_image(images_2)
                i_feats_2 = image_feats_2[:, 0, :].float()
                loss_list = [
                    objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale),
                    objectives.compute_sdm(i_feats_2, t_feats, batch['pids'], logit_scale),
                    objectives.compute_sdm(i_feats_2, i_feats, batch['pids'], logit_scale)
                ]
                sdm_loss = 0
                for item in loss_list:
                    sdm_loss += item * (1/len(loss_list))
                ret.update({'sdm_loss':sdm_loss})
            else:
                ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mim' in self.current_task:
            images_origin = batch['images_origin']
            with torch.no_grad():
                images_discrete,bool_masked_pos,mim_labels = self.get_codebook_indices(images_origin)
                mim_labels = mim_labels.reshape(-1)
            batch_size,seq_len = image_feats.shape[0],image_feats.shape[1]
            image_feats_masked = image_feats*~bool_masked_pos.unsqueeze(-1) + self.image_mask_token.expand(batch_size,seq_len,-1)*bool_masked_pos.unsqueeze(-1)
            # print(image_feats_masked)
            #print(text_feats.float().dtype)
            # print(text_feats)
            #.to(torch.float32)
            x = self.cross_former(image_feats_masked.to(image_feats.dtype), text_feats.to(image_feats.dtype), text_feats.to(image_feats.dtype))
            x = self.mim_head(x)
            scores = x.float().reshape(-1, self.args.dalle_vocab_size)
            ret.update({'mim_loss': objectives.compute_mim(scores, mim_labels)*self.args.mim_loss_weight})

            pred = scores.max(1)[1]
            mim_label_idx = torch.nonzero(mim_labels!=-100)
            acc = (pred[mim_label_idx] == mim_labels[mim_label_idx]).float().mean()
            ret.update({'mim_acc': acc})
            pass
        
        if "mim_part" in self.current_task:
            image_feats_masked,img_feature_gt,img_token_mask_id = self.base_model.make_image_mask(
                images,batch['label'],self.args.image_bck_mask_prob,self.args.image_part_mask_prob,self.mim_part_mask_token
            )
            feature_dim = image_feats.shape[-1]
            x = self.cross_former_image(
                image_feats_masked.to(self.base_model.dtype), 
                text_feats.to(self.base_model.dtype), 
                text_feats.to(self.base_model.dtype)
            )
            x = x.reshape(-1,feature_dim)[img_token_mask_id.reshape(-1)]
            x = self.mim_decoder(x)
            ret.update({'mim_part_loss': self.mim_loss(x,img_feature_gt)*self.args.mim_loss_weight})
            

        # if "mim_part" in self.current_task:
        #     batchsize = image_feats.shape[0]
        #     feature_dim = image_feats.shape[-1]
        #     cls_labels = torch.ones((batchsize,1)).bool()
        #     seq_len = batch['label'].shape[1]
        #     part_mask = torch.cat([~cls_labels.to(images.device),(torch.randn((batchsize,seq_len))<self.args.image_part_mask_prob).to(images.device) * batch['label'].bool()],dim=1) 
        #     bck_mask = torch.cat([~cls_labels.to(images.device),(torch.randn((batchsize,seq_len))<self.args.image_bck_mask_prob).to(images.device) * ~batch['label'].bool()],dim=1)
        #     img_token_mask = (part_mask+bck_mask).reshape(-1)
        #     img_token_mask_id = torch.nonzero(img_token_mask)
        #     img_feature_gt = image_feats.reshape(-1,feature_dim)[img_token_mask_id.reshape(-1)]

        #     image_feats_masked = image_feats.reshape(-1,feature_dim)*~img_token_mask.unsqueeze(-1) + self.mim_part_mask_token.expand(batchsize*(seq_len+1),-1)*img_token_mask.unsqueeze(-1)
        #     image_feats_masked = image_feats_masked.reshape(batchsize,(seq_len+1),-1)
            
        #     x = self.cross_former_image(
        #         image_feats_masked.to(image_feats.dtype), 
        #         text_feats.to(image_feats.dtype), 
        #         text_feats.to(image_feats.dtype)
        #     )
        #     x = x.reshape(-1,feature_dim)[img_token_mask_id.reshape(-1)]
        #     x = self.mim_decoder(x)
            
        #     ret.update({'mim_part_loss': self.mim_loss(x,img_feature_gt)*self.args.mim_loss_weight})
        #     pass

        if "matching" in self.current_task:
            batchsize = image_feats.shape[0]
            attr_index = batch['attr_index']
            part_num = batch['part_num']
            img_labels = batch['label']
            match_matrix = batch['pids'].reshape(-1,1).ne(batch['pids'].reshape(1,-1))
            #loss_attr = 0
            img_part_pos = []
            text_part_pos = []
            pos_list = []
            for i in range(batchsize):
                atr_i = attr_index[i,:part_num[i]]
                img_part_feats = []
                txt_attr_feats = []
                for j in range(part_num[i]):
                    ignore_index = (img_labels[i]==(j+1)).sum()==0 or atr_i[j][0]==-1
                    if ignore_index:
                        continue
                    # txt_attr_feats.append(text_feats[i][atr_i[j][0]:atr_i[j][1]+1].mean(dim=0))
                    # img_part_feats.append(image_feats[i,1:][img_labels[i]==(j+1)].mean(dim=0))
                    img_part_feats.append(torch.cat([image_feats[i,:1],image_feats[i,1:][img_labels[i]==(j+1)]]).unsqueeze(0))
                    txt_attr_feats.append(text_feats[i][atr_i[j][0]:atr_i[j][1]+1].unsqueeze(0))
                    x_output,x_score = self.cross_attn_matching(
                        torch.cat([image_feats[i,:1],image_feats[i,1:][img_labels[i]==(j+1)]]).unsqueeze(0),
                        text_feats[i][atr_i[j][0]:atr_i[j][1]+1].unsqueeze(0),
                        text_feats[i][atr_i[j][0]:atr_i[j][1]+1].unsqueeze(0)
                    )
                    pos_list.append(self.classifier_matching(x_output[0][0]))
                    pass
                img_part_pos.append(img_part_feats)
                text_part_pos.append(txt_attr_feats)

            non_zero_list = torch.Tensor([len(i)>0 for i in img_part_pos]).bool().to(match_matrix.device)
            non_zero_list = non_zero_list.reshape(-1,1) * non_zero_list
            neg_list = []
            for i in range(batchsize):
                non_zero_index = torch.nonzero(match_matrix[i]*non_zero_list[i])
                txt_neg_feats = []
                if len(non_zero_index) == 0:
                    continue
                for j in range(len(img_part_pos[i])):
                    x = non_zero_index[torch.randint(0,len(non_zero_index),(1,1))]
                    y = torch.randint(0,len(text_part_pos[x]),(1,1))
                    txt_neg_feats.append(text_part_pos[x][y])
                    x_output,x_score = self.cross_attn_matching(
                        img_part_pos[i][j],
                        text_part_pos[x][y],
                        text_part_pos[x][y]
                    )
                    neg_list.append(self.classifier_matching(x_output[0][0]))
                
                txt_neg_feats.append(txt_neg_feats)

            pos_label = torch.ones(len(pos_list),dtype=torch.long)
            neg_label = torch.zeros(len(neg_list),dtype=torch.long)
            pred = torch.stack(pos_list+neg_list)
            label = torch.cat([pos_label,neg_label]).to(pred.device)

            ret.update({'matching_loss': F.cross_entropy(pred,label) })
            pass

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if 'mlm_part' in self.current_task:
            mlm_ids = batch['mlm_part_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_part_labels'].reshape(-1)
            ret.update({'mlm_part_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    if args.using_fp32 == False:
        print("Running convert weights to fp16!")
        convert_weights(model)
    return model
