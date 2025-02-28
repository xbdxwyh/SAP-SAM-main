import torch
import numpy as np
import logging

import utils
import time
import datetime

import torch.nn.functional as F


@torch.no_grad()
#def evaluation(model, data_loader, tokenizer, device, config):
def evaluation(model, img_dataloader, txt_dataloader, device, config,fast = False):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    #texts = data_loader.dataset.text   
    # num_text = len(txt_dataloader.dataset)
    # text_bs = 256
    text_feats = []
    text_embeds = []  
    text_atts = []
    text_labels = []
    for i, [text_item, caption_matching_img_index] in enumerate(txt_dataloader):
        label = text_item['label']
        input_ids = text_item['input_ids'].to(device)
        token_type_ids = text_item['token_type_ids'].to(device)
        attention_mask = text_item['attention_mask'].to(device)

        #text_output = model.text_encoder(input_ids, attention_mask = attention_mask, mode='text')  

        text_output = model.text_prompt_encoder(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            using_prompt = True
        )

        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_feat_proj(text_feat[:,0,:]))
        text_embeds.append(text_embed)   
        text_feats.append(text_feat)
        text_atts.append(attention_mask)

    text_embeds = torch.cat(text_embeds,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_labels = torch.cat(text_labels).view(-1)
    
    image_feats = []
    image_embeds = []
    image_labels = []
    for times, image_item in enumerate(img_dataloader):
        pixel_values = image_item['pixel_values'].to(device)
        #label = label.to(device)

        image_labels.append(label)
    #for image, img_id in data_loader: 
        #image = image.to(device) 
        #image_feat = model.visual_encoder(image)      

        image_outputs = model.forward_image_encoder(
            pixel_values=pixel_values,
        )

        image_hidden_states = image_outputs.last_hidden_state
        batch_size = image_outputs.shape[0]
        hidden_size = image_outputs.shape[-1]

        image_feat = image_hidden_states.view(batch_size,-1,hidden_size)
        image_embeddings = image_hidden_states.mean(dim=1)  
        image_embed = model.vision_feat_proj(image_embeddings)            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat)
        image_embeds.append(image_embed)
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    image_labels = torch.cat(image_labels).view(-1)
    
    sims_matrix = image_embeds @ text_embeds.t()
    # evaluate none last fusion
    ############################
    t2i_cmc, t2i_map = evaluate(sims_matrix.cpu().t(), text_labels, image_labels)
    results_before = {
        "t2i @R1 : ":t2i_cmc[0]*100.0,
        "t2i @R5 : ":t2i_cmc[4]*100.0,
        "t2i @R10 : ":t2i_cmc[9]*100.0,
        #"t2i @mAP : ":t2i_map*100.0,
    }

    if fast:
        results_t2i = {
            "t2i @R1 : ":0*100.0,
            "t2i @R5 : ":0*100.0,
            "t2i @R10 : ":0*100.0,
            #"t2i @mAP : ":0*100.0,
        }
        results_i2t = {
            "i2t @R1 : ":0*100.0,
            "i2t @R5 : ":0*100.0,
            "i2t @R10 : ":0*100.0,
            #"i2t @mAP : ":0*100.0,
        }
        return results_before,results_t2i,results_i2t

    score_matrix_i2t = torch.full((len(img_dataloader.dataset),len(txt_dataloader.dataset)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    # for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
    #     topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

    #     encoder_output = image_feats[start+i].repeat(config['k_test'],1,1)
    #     encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
    #     output = model.text_encoder(encoder_embeds = text_feats[topk_idx], 
    #                                 attention_mask = text_atts[topk_idx],
    #                                 encoder_hidden_states = encoder_output,
    #                                 encoder_attention_mask = encoder_att,                             
    #                                 return_dict = True,
    #                                 mode = 'fusion'
    #                                )
    #     score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
    #     score_matrix_i2t[start+i,topk_idx] = score
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(txt_dataloader.dataset),len(img_dataloader.dataset)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.cross_attention(
            hidden_states = text_feats[start+i].repeat(config['k_test'],1,1), 
            attention_mask = text_atts[start+i].repeat(config['k_test'],1)[:, None, None, :],
            encoder_hidden_states = encoder_output,
            encoder_attention_mask = encoder_att[:, None, None, :],                             
            return_dict = True,
        )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score

    # if args.distributed:
    #     dist.barrier()   
    #     torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
    #     torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    t2i_cmc, t2i_map = evaluate(torch.Tensor(score_matrix_t2i.cpu().numpy()), text_labels, image_labels)
    results_t2i = {
        "t2i @R1 : ":t2i_cmc[0]*100.0,
        "t2i @R5 : ":t2i_cmc[4]*100.0,
        "t2i @R10 : ":t2i_cmc[9]*100.0,
        #"t2i @mAP : ":t2i_map*100.0,
    }
    # i2t_cmc, i2t_map = evaluate(torch.Tensor(score_matrix_i2t.cpu().numpy()).t(), text_labels, image_labels)
    # results_i2t = {
    #     "i2t @R1 : ":i2t_cmc[0]*100.0,
    #     "i2t @R5 : ":i2t_cmc[4]*100.0,
    #     "i2t @R10 : ":i2t_cmc[9]*100.0,
    #     "i2t @mAP : ":i2t_map*100.0,
    # }
    results_i2t = {
        "i2t @R1 : ":0*100.0,
        "i2t @R5 : ":0*100.0,
        "i2t @R10 : ":0*100.0,
        #"i2t @mAP : ":0*100.0,
    }
    return results_before,results_t2i,results_i2t
    #return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(),text_labels,image_labels



@torch.no_grad()
#def evaluation(model, data_loader, tokenizer, device, config):
def evaluation_clip(model, img_dataloader, txt_dataloader, device, config,fast = False):
    # test
    model.eval() 
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    print('Computing features for evaluation...')
    start_time = time.time()  
    #texts = data_loader.dataset.text   
    # num_text = len(txt_dataloader.dataset)
    # text_bs = 256
    text_feats = []
    text_embeds = []  
    text_atts = []
    text_labels = []
    for i, [label, input_ids, caption_length, attention_mask, caption_matching_img_index] in enumerate(txt_dataloader):
        #label = label.to(device)
        text_labels.append(label)
        input_ids = input_ids.to(device).long()
        caption_length = caption_length.to(device)
        attention_mask = attention_mask.to(device)
    #for i in range(0, num_text, text_bs):
        #text = texts[i: min(num_text, i+text_bs)]
        #text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
        text_output = model.text_encoder(input_ids, attention_mask = attention_mask)  
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        text_embeds.append(text_embed) 
        text_feat = model.text_feat_proj(text_feat)
        text_feats.append(text_feat)
        text_atts.append(attention_mask)

    text_embeds = torch.cat(text_embeds,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_labels = torch.cat(text_labels).view(-1)
    
    image_feats = []
    image_embeds = []
    image_labels = []
    for times, [image, label] in enumerate(img_dataloader):
        image = image.to(device)
        #label = label.to(device)
        image_labels.append(label)
    #for image, img_id in data_loader: 
        #image = image.to(device) 
        image_output = model.visual_encoder(image)  
        image_feat = image_output.last_hidden_state
        #image_feat = model.visual_encoder(image)        
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat)
        image_embeds.append(image_embed)
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    image_labels = torch.cat(image_labels).view(-1)
    
    sims_matrix = image_embeds @ text_embeds.t()
    # evaluate none last fusion
    ############################
    t2i_cmc, t2i_map = evaluate(sims_matrix.cpu().t(), text_labels, image_labels)
    results_before = {
        "t2i @R1 : ":t2i_cmc[0]*100.0,
        "t2i @R5 : ":t2i_cmc[4]*100.0,
        "t2i @R10 : ":t2i_cmc[9]*100.0,
        #"t2i @mAP : ":t2i_map*100.0,
    }

    if fast:
        results_t2i = {
            "t2i @R1 : ":0*100.0,
            "t2i @R5 : ":0*100.0,
            "t2i @R10 : ":0*100.0,
            #"t2i @mAP : ":0*100.0,
        }
        results_i2t = {
            "i2t @R1 : ":0*100.0,
            "i2t @R5 : ":0*100.0,
            "i2t @R10 : ":0*100.0,
            #"i2t @mAP : ":0*100.0,
        }
        return results_before,results_t2i,results_i2t

    score_matrix_i2t = torch.full((len(img_dataloader.dataset),len(txt_dataloader.dataset)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    # for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
    #     topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

    #     encoder_output = image_feats[start+i].repeat(config['k_test'],1,1)
    #     encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
    #     output = model.text_encoder(encoder_embeds = text_feats[topk_idx], 
    #                                 attention_mask = text_atts[topk_idx],
    #                                 encoder_hidden_states = encoder_output,
    #                                 encoder_attention_mask = encoder_att,                             
    #                                 return_dict = True,
    #                                 mode = 'fusion'
    #                                )
    #     score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
    #     score_matrix_i2t[start+i,topk_idx] = score
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(txt_dataloader.dataset),len(img_dataloader.dataset)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.multi_modal_encoder(
            encoder_embeds = text_feats[start+i].repeat(config['k_test'],1,1), 
            attention_mask = text_atts[start+i].repeat(config['k_test'],1),
            encoder_hidden_states = encoder_output,
            encoder_attention_mask = encoder_att,                             
            return_dict = True,
            mode = 'fusion'
        )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score

    # if args.distributed:
    #     dist.barrier()   
    #     torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
    #     torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    t2i_cmc, t2i_map = evaluate(torch.Tensor(score_matrix_t2i.cpu().numpy()), text_labels, image_labels)
    results_t2i = {
        "t2i @R1 : ":t2i_cmc[0]*100.0,
        "t2i @R5 : ":t2i_cmc[4]*100.0,
        "t2i @R10 : ":t2i_cmc[9]*100.0,
        #"t2i @mAP : ":t2i_map*100.0,
    }
    # i2t_cmc, i2t_map = evaluate(torch.Tensor(score_matrix_i2t.cpu().numpy()).t(), text_labels, image_labels)
    # results_i2t = {
    #     "i2t @R1 : ":i2t_cmc[0]*100.0,
    #     "i2t @R5 : ":i2t_cmc[4]*100.0,
    #     "i2t @R10 : ":i2t_cmc[9]*100.0,
    #     "i2t @mAP : ":i2t_map*100.0,
    # }
    results_i2t = {
        "i2t @R1 : ":0*100.0,
        "i2t @R5 : ":0*100.0,
        "i2t @R10 : ":0*100.0,
        #"i2t @mAP : ":0*100.0,
    }
    return results_before,results_t2i,results_i2t
    #return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(),text_labels,image_labels




def jaccard(a_list, b_list):
    return float(len(set(a_list) & set(b_list))) / float(len(set(a_list) | set(b_list)))


def jaccard_mat(row_nn, col_nn):
    jaccard_sim = np.zeros((row_nn.shape[0], col_nn.shape[0]))
    # FIXME: need optimization
    for i in range(row_nn.shape[0]):
        for j in range(col_nn.shape[0]):
            jaccard_sim[i, j] = jaccard(row_nn[i], col_nn[j])
    return torch.from_numpy(jaccard_sim)



def k_reciprocal(q_feats, g_feats, neighbor_num=5, alpha=0.05):
    qg_sim = torch.matmul(q_feats, g_feats.t())  # q * g
    gg_sim = torch.matmul(g_feats, g_feats.t())  # g * g

    qg_indices = torch.argsort(qg_sim, dim=1, descending=True)
    gg_indices = torch.argsort(gg_sim, dim=1, descending=True)

    qg_nn = qg_indices[:, :neighbor_num]  # q * n
    gg_nn = gg_indices[:, :neighbor_num]  # g * n

    jaccard_sim = jaccard_mat(qg_nn.cpu().numpy(), gg_nn.cpu().numpy())  # q * g
    jaccard_sim = jaccard_sim.to(qg_sim.device)
    return alpha * jaccard_sim  # q * g




@torch.no_grad()
#def evaluation(model, data_loader, tokenizer, device, config):
def evaluation_with_rerank(model, img_dataloader, txt_dataloader, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    #texts = data_loader.dataset.text   
    # num_text = len(txt_dataloader.dataset)
    # text_bs = 256
    text_feats = []
    text_embeds = []  
    text_atts = []
    text_labels = []
    for i, [label, input_ids, caption_length, attention_mask, _] in enumerate(txt_dataloader):
        #label = label.to(device)
        text_labels.append(label)
        input_ids = input_ids.to(device).long()
        caption_length = caption_length.to(device)
        attention_mask = attention_mask.to(device)
    #for i in range(0, num_text, text_bs):
        #text = texts[i: min(num_text, i+text_bs)]
        #text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
        text_output = model.text_encoder(input_ids, attention_mask = attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        text_embeds.append(text_embed)   
        text_feats.append(text_feat)
        text_atts.append(attention_mask)

    text_embeds = torch.cat(text_embeds,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_labels = torch.cat(text_labels).view(-1)
    
    image_feats = []
    image_embeds = []
    image_labels = []
    for times, [image, label] in enumerate(img_dataloader):
        image = image.to(device)
        #label = label.to(device)
        image_labels.append(label)
    #for image, img_id in data_loader: 
        #image = image.to(device) 
        image_feat = model.visual_encoder(image)        
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat)
        image_embeds.append(image_embed)
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    image_labels = torch.cat(image_labels).view(-1)
    
    sims_matrix = image_embeds @ text_embeds.t()

    rerank_sim_matrix = k_reciprocal(image_embeds,text_embeds)
    sims_matrix_ = sims_matrix + rerank_sim_matrix

    # evaluate none last fusion
    ############################
    # t2i_cmc, t2i_map = evaluate(sims_matrix.cpu().t(), text_labels, image_labels)
    # results_before = {
    #     "t2i @R1 : ":t2i_cmc[0]*100.0,
    #     "t2i @R5 : ":t2i_cmc[4]*100.0,
    #     "t2i @R10 : ":t2i_cmc[9]*100.0,
    #     "t2i @mAP : ":t2i_map*100.0,
    # }
    score_matrix_i2t = torch.full((len(img_dataloader.dataset),len(txt_dataloader.dataset)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix_.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix_.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix_[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start+i].repeat(config['k_test'],1,1)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[topk_idx], 
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score
        
    sims_matrix = sims_matrix.t()
    rerank_sim_matrix_ = k_reciprocal(text_embeds,image_embeds)
    sims_matrix_ = sims_matrix + rerank_sim_matrix_

    
    score_matrix_t2i = torch.full((len(txt_dataloader.dataset),len(img_dataloader.dataset)),-100.0).to(device)
    
    step = sims_matrix_.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix_.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix_[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[start+i].repeat(config['k_test'],1,1), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score

    # if config['distributed']:
    #     dist.barrier()   
    #     torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
    #     torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 
    return torch.Tensor(score_matrix_t2i.cpu().numpy()),torch.Tensor(score_matrix_i2t.cpu().numpy()).t(),\
        rerank_sim_matrix_,rerank_sim_matrix, text_labels, image_labels

    t2i_cmc, t2i_map = evaluate(torch.Tensor(score_matrix_t2i.cpu().numpy()) + rerank_sim_matrix_, text_labels, image_labels)
    results_t2i = {
        "rerank t2i @R1 : ":t2i_cmc[0]*100.0,
        "rerank t2i @R5 : ":t2i_cmc[4]*100.0,
        "rerank t2i @R10 : ":t2i_cmc[9]*100.0,
        #"rerank t2i @mAP : ":t2i_map*100.0,
    }
    i2t_cmc, i2t_map = evaluate(torch.Tensor(score_matrix_i2t.cpu().numpy()).t() + rerank_sim_matrix, text_labels, image_labels)
    results_i2t = {
        "rerank i2t @R1 : ":i2t_cmc[0]*100.0,
        "rerank i2t @R5 : ":i2t_cmc[4]*100.0,
        "rerank i2t @R10 : ":i2t_cmc[9]*100.0,
        #"rerank i2t @mAP : ":i2t_map*100.0,
    }
    return results_t2i,results_i2t#,results_t2i_rerank,results_i2t_rerank


def calculate_ap(similarity, label_query, label_gallery):
    """
        calculate the similarity, and rank the distance, according to the distance, calculate the ap, cmc
    :param label_query: the id of query [1]
    :param label_gallery:the id of gallery [N]
    :return: ap, cmc
    """

    index = np.argsort(similarity)[::-1]  # the index of the similarity from huge to small
    good_index = np.argwhere(label_gallery == label_query)  # the index of the same label in gallery

    cmc = np.zeros(index.shape)

    mask = np.in1d(index, good_index)  # get the flag the if index[i] is in the good_index

    precision_result = np.argwhere(mask == True)  # get the situation of the good_index in the index

    precision_result = precision_result.reshape(precision_result.shape[0])

    if precision_result.shape[0] != 0:
        cmc[int(precision_result[0]):] = 1  # get the cmc

        d_recall = 1.0 / len(precision_result)
        ap = 0

        for i in range(len(precision_result)):  # ap is to calculate the PR area
            precision = (i + 1) * 1.0 / (precision_result[i] + 1)

            if precision_result[i] != 0:
                old_precision = i * 1.0 / precision_result[i]
            else:
                old_precision = 1.0

            ap += d_recall * (old_precision + precision) / 2

        return ap, cmc
    else:
        return None, None

@torch.no_grad()
def evaluate(similarity, label_query, label_gallery):
    similarity = similarity.numpy()
    label_query = label_query.numpy()
    label_gallery = label_gallery.numpy()

    cmc = np.zeros(label_gallery.shape)
    ap = 0
    for i in range(len(label_query)):
        ap_i, cmc_i = calculate_ap(similarity[i, :], label_query[i], label_gallery)
        cmc += cmc_i
        ap += ap_i
    """
    cmc_i is the vector [0,0,...1,1,..1], the first 1 is the first right prediction n,
    rank-n and the rank-k after it all add one right prediction, therefore all of them's index mark 1
    Through the  add all the vector and then divive the n_query, we can get the rank-k accuracy cmc
    cmc[k-1] is the rank-k accuracy   
    """
    cmc = cmc / len(label_query)
    map = ap / len(label_query)  # map = sum(ap) / n_query

    # print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (cmc[0], cmc[4], cmc[9], map))

    return cmc, map


@torch.no_grad()
def evaluate_mmtbpr_sam(epoch, iteration, model, img_dataloader, txt_dataloader,albef_config, device, dataset_name,fast=False,model_type="albef"):
    logger = logging.getLogger("PersonSearch.inference")
    #txt_root = os.path.join(opt.save_path, 'log', 'test_separate.log')
    #best_txt_root = os.path.join(opt.save_path, 'log', 'best_test.log')

    logger.info(
        "Start evaluation on {} dataset({} images).".format(dataset_name, len(img_dataloader))
    )

    logger.info(
        "Epoch {}, iteration {}.".format(epoch, iteration)
    )

    results_before,results_t2i,results_i2t = evaluation(model,img_dataloader,txt_dataloader,device,albef_config,fast)
    
    logger.info(results_before)
    logger.info(results_t2i)
    logger.info(results_i2t)

    # best = write_result(similarity, img_labels, txt_labels, txt_img_index, 'similarity_all:',
    #                     txt_root, best_txt_root, epoch, best, iteration)
    if fast:
        return results_before['t2i @R1 : ']
    else:
        return results_t2i['t2i @R1 : ']


