import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_sdm_soft(
        image_features, 
        text_features, 
        image_features_s,
        text_features_s,
        pid, 
        logit_scale, 
        image_id=None, 
        factor=0.3, 
        epsilon=1e-8,
        alpha=0.4
    ):
    """
    Similarity Distribution Matching
    """
    pid = pid.reshape((-1, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    sim_targets = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        sim_targets = (sim_targets - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    # normalize the true matching distribution
    sim_targets = sim_targets / sim_targets.sum(dim=1)

    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)

    i_feats_s_norm = F.normalize(image_features_s)
    t_feats_s_norm = F.normalize(text_features_s)

    with torch.no_grad():
        sim_i2t_s = i_feats_s_norm @ t_feats_s_norm.t()
        sim_t2i_s = t_feats_s_norm @ i_feats_s_norm.t()
        sim_i2t_targets = alpha * F.softmax(sim_i2t_s, dim=1) + (1 - alpha) * sim_targets
        sim_t2i_targets = alpha * F.softmax(sim_t2i_s, dim=1) + (1 - alpha) * sim_targets

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(sim_i2t_targets + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(sim_t2i_targets + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_sdm_soft_queue(
        image_features, 
        text_features, 
        image_features_s,
        text_features_s,
        image_features_all,
        text_features_all,
        pid,
        pid_all, 
        logit_scale, 
        epsilon=1e-8,
        alpha=0.4
    ):
    """
    Similarity Distribution Matching
    """
    pos_idx = torch.eq(pid.reshape((-1, 1)), pid_all.reshape(1,-1)).float()  
    sim_targets = pos_idx / pos_idx.sum(1,keepdim=True) 

    pos_idx_online = torch.eq(pid.reshape((-1, 1)), pid.reshape(1,-1)).float()  
    sim_targets_online = pos_idx_online / pos_idx_online.sum(1,keepdim=True) 

    with torch.no_grad():
        sim_i2t_s = logit_scale * image_features_s @ text_features_all.t()
        sim_t2i_s = logit_scale * text_features_s @ image_features_all.t()
        sim_i2t_targets = alpha * F.softmax(sim_i2t_s, dim=1) + (1 - alpha) * sim_targets
        sim_t2i_targets = alpha * F.softmax(sim_t2i_s, dim=1) + (1 - alpha) * sim_targets
        
        sim_i2t_online = logit_scale * image_features_s @ text_features_s.t()
        sim_t2i_online = logit_scale * text_features_s @ image_features_s.t()

        sim_i2t_targets_online = alpha * F.softmax(sim_i2t_online, dim=1) + (1 - alpha) * sim_targets_online
        sim_t2i_targets_online = alpha * F.softmax(sim_t2i_online, dim=1) + (1 - alpha) * sim_targets_online

    t2i_cosine_theta = text_features @ image_features_all.t()
    i2t_cosine_theta = image_features @ text_features_all.t()

    t2i_cosine_theta_online = text_features @ image_features.t()
    i2t_cosine_theta_online = image_features @ text_features.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    text_proj_image_online = logit_scale * t2i_cosine_theta_online
    image_proj_text_online = logit_scale * i2t_cosine_theta_online

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(sim_i2t_targets + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(sim_t2i_targets + epsilon))

    i2t_pred_online = F.softmax(image_proj_text_online, dim=1)
    i2t_loss_online = i2t_pred_online * (F.log_softmax(image_proj_text_online, dim=1) - torch.log(sim_i2t_targets_online + epsilon))
    t2i_pred_online = F.softmax(text_proj_image_online, dim=1)
    t2i_loss_online = t2i_pred_online * (F.log_softmax(text_proj_image_online, dim=1) - torch.log(sim_t2i_targets_online + epsilon))


    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1)) +\
            torch.mean(torch.sum(i2t_loss_online, dim=1)) + torch.mean(torch.sum(t2i_loss_online, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)

def compute_mim(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    return ce(scores, labels)

def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)
    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss

def compute_n_itc(image_features, text_features, image_features_s, text_features_s, pid, logit_scale, alpha=0.3):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    pid = pid.view(-1, 1)
    pid_all = pid.view(1, -1)
    pos_idx = torch.eq(pid, pid_all).float()
    sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
    sim_targets = sim_targets.to(image_features.device)

    i_feats_norm = F.normalize(image_features)
    t_feats_norm = F.normalize(text_features)

    i_feats_s_norm = F.normalize(image_features_s)
    t_feats_s_norm = F.normalize(text_features_s)

    with torch.no_grad():
        sim_i2t_s = logit_scale * i_feats_s_norm @ t_feats_s_norm.t()
        sim_t2i_s = logit_scale * t_feats_s_norm @ i_feats_s_norm.t()
        sim_i2t_targets = alpha * F.softmax(sim_i2t_s, dim=1) + (1 - alpha) * sim_targets
        sim_t2i_targets = alpha * F.softmax(sim_t2i_s, dim=1) + (1 - alpha) * sim_targets

    sim_i2t = logit_scale * i_feats_norm @ t_feats_norm.t()
    sim_t2i = logit_scale * t_feats_norm @ i_feats_norm.t()

    loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
    loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
    loss_ita = (loss_i2t + loss_t2i) / 2

    return loss_ita


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

