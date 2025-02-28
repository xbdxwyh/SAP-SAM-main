import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).data.cpu(), 1
        )
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def instance_loss(
    projection, visual_embed, textual_embed, labels, scale=1, norm=False, epsilon=0.0
):
    if norm:
        visual_norm = F.normalize(visual_embed, p=2, dim=-1)
        textual_norm = F.normalize(textual_embed, p=2, dim=-1)
    else:
        visual_norm = visual_embed
        textual_norm = textual_embed
    projection_norm = F.normalize(projection, p=2, dim=0)

    visual_logits = scale * torch.matmul(visual_norm, projection_norm)
    textual_logits = scale * torch.matmul(textual_norm, projection_norm)

    # if epsilon > 0:
    #     criterion = CrossEntropyLabelSmooth(num_classes=projection_norm.shape[1])
    # else:
    criterion = nn.CrossEntropyLoss(reduction="mean",label_smoothing=epsilon)
    loss = criterion(visual_logits, labels) + criterion(textual_logits, labels)

    return loss


def cmpc_loss(projection, visual_embed, textual_embed, labels, verbose=False):
    """
    Cross-Modal Projection Classfication loss (CMPC)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
    """
    visual_norm = F.normalize(visual_embed, p=2, dim=1)
    textual_norm = F.normalize(textual_embed, p=2, dim=1)
    projection_norm = F.normalize(projection, p=2, dim=0)

    image_proj_text = (
        torch.sum(visual_embed * textual_norm, dim=1, keepdim=True) * textual_norm
    )
    text_proj_image = (
        torch.sum(textual_embed * visual_norm, dim=1, keepdim=True) * visual_norm
    )

    image_logits = torch.matmul(image_proj_text, projection_norm)
    text_logits = torch.matmul(text_proj_image, projection_norm)

    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(image_logits, labels) + criterion(text_logits, labels)

    # classification accuracy for observation
    if verbose:
        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_precision = torch.mean((image_pred == labels).float())
        text_precision = torch.mean((text_pred == labels).float())

        return loss, image_precision, text_precision
    return loss


def global_align_loss(
    visual_embed,
    textual_embed,
    labels,
    alpha=0.6,
    beta=0.4,
    scale_pos=10,
    scale_neg=40,
):
    batch_size = labels.size(0)
    visual_norm = F.normalize(visual_embed, p=2, dim=1)
    textual_norm = F.normalize(textual_embed, p=2, dim=1)
    similarity = torch.matmul(visual_norm, textual_norm.t())
    labels_ = (
        labels.expand(batch_size, batch_size)
        .eq(labels.expand(batch_size, batch_size).t())
        .float()
    )

    pos_inds = labels_ == 1
    neg_inds = labels_ == 0
    loss_pos = torch.log(1 + torch.exp(-scale_pos * (similarity[pos_inds] - alpha)))
    loss_neg = torch.log(1 + torch.exp(scale_neg * (similarity[neg_inds] - beta)))
    loss = (loss_pos.sum() + loss_neg.sum()) * 2.0

    loss /= batch_size
    return loss


def global_align_loss_from_sim(
    similarity,
    labels,
    alpha=0.6,
    beta=0.4,
    scale_pos=10,
    scale_neg=40,
):
    batch_size = labels.size(0)
    labels_ = (
        labels.expand(batch_size, batch_size)
        .eq(labels.expand(batch_size, batch_size).t())
        .float()
    )

    pos_inds = labels_ == 1
    neg_inds = labels_ == 0
    loss_pos = torch.log(1 + torch.exp(-scale_pos * (similarity[pos_inds] - alpha)))
    loss_neg = torch.log(1 + torch.exp(scale_neg * (similarity[neg_inds] - beta)))
    loss = (loss_pos.sum() + loss_neg.sum()) * 2.0

    loss /= batch_size
    return loss


def cmpm_loss(visual_embed, textual_embed, labels, verbose=False, epsilon=1e-8):
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

    batch_size = visual_embed.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = labels_dist == 0

    visual_norm = F.normalize(visual_embed, p=2, dim=1)
    textual_norm = F.normalize(textual_embed, p=2, dim=1)
    image_proj_text = torch.matmul(visual_embed, textual_norm.t())
    text_proj_image = torch.matmul(textual_embed, visual_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (
        F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon)
    )

    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (
        F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon)
    )

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(
        torch.sum(t2i_loss, dim=1)
    )

    if verbose:
        sim_cos = torch.matmul(visual_norm, textual_norm.t())

        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))

        return loss, pos_avg_sim, neg_avg_sim
    return loss


def infonce_loss(
    v_pos,
    v_neg,
    t_pos,
    t_neg,
    T=0.07,
):
    v_logits = torch.cat([v_pos, v_neg], dim=1) / T
    t_logits = torch.cat([t_pos, t_neg], dim=1) / T
    labels = torch.zeros(v_logits.shape[0], dtype=torch.long).cuda()
    loss = F.cross_entropy(v_logits, labels) + F.cross_entropy(t_logits, labels)
    return loss



def compute_ranking_loss(
    similarity,
    labels,
    margin = 0.1
):
    def semi_hard_negative(loss):
        negative_index = np.where(np.logical_and(loss < margin, loss > 0))[0]
        return np.random.choice(negative_index) if len(negative_index) > 0 else None

    def get_triplets(similarity, labels):
        similarity = similarity.cpu().data.numpy()

        labels = labels.cpu().data.numpy()
        triplets = []

        for idx, label in enumerate(labels):  # same class calculate together

            negative = np.where(labels != label)[0]

            ap_sim = similarity[idx, idx]
            # print(ap_combination_list.shape, ap_distances_list.shape)

            loss = similarity[idx, negative] - ap_sim + margin

            negetive_index = semi_hard_negative(loss)

            if negetive_index is not None:
                triplets.append([idx, idx, negative[negetive_index]])

        if len(triplets) == 0:
            triplets.append([idx, idx, negative[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)
    
    image_triplets = get_triplets(similarity, labels)
    text_triplets = get_triplets(similarity.t(), labels)

    # print(image_triplets.size(), text_triplets.size())
    image_anchor_loss = F.relu(margin
                        - similarity[image_triplets[:, 0], image_triplets[:, 1]]
                        + similarity[image_triplets[:, 0], image_triplets[:, 2]])

    texy_anchor_loss = F.relu(margin
                        - similarity[text_triplets[:, 0], text_triplets[:, 1]]
                        + similarity[text_triplets[:, 0], text_triplets[:, 2]])

    loss = torch.sum(image_anchor_loss) + torch.sum(texy_anchor_loss)
    # loss = CMPM_loss + CMPC_loss

    return loss
    
    pass


class RankingLoss(nn.Module):

    def __init__(self, margin):
        super(RankingLoss, self).__init__()
        self.margin = margin
        #self.device = opt.device

    def semi_hard_negative(self, loss):
        negative_index = np.where(np.logical_and(loss < self.margin, loss > 0))[0]
        return np.random.choice(negative_index) if len(negative_index) > 0 else None

    def get_triplets(self, similarity, labels):
        similarity = similarity.cpu().data.numpy()

        labels = labels.cpu().data.numpy()
        triplets = []

        for idx, label in enumerate(labels):  # same class calculate together

            negative = np.where(labels != label)[0]

            ap_sim = similarity[idx, idx]
            # print(ap_combination_list.shape, ap_distances_list.shape)

            loss = similarity[idx, negative] - ap_sim + self.margin

            negetive_index = self.semi_hard_negative(loss)

            if negetive_index is not None:
                triplets.append([idx, idx, negative[negetive_index]])

        if len(triplets) == 0:
            triplets.append([idx, idx, negative[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

    def forward(self, similarity, label):

        image_triplets = self.get_triplets(similarity, label)
        text_triplets = self.get_triplets(similarity.t(), label)

        # print(image_triplets.size(), text_triplets.size())
        image_anchor_loss = F.relu(self.margin
                            - similarity[image_triplets[:, 0], image_triplets[:, 1]]
                            + similarity[image_triplets[:, 0], image_triplets[:, 2]])

        texy_anchor_loss = F.relu(self.margin
                            - similarity[text_triplets[:, 0], text_triplets[:, 1]]
                            + similarity[text_triplets[:, 0], text_triplets[:, 2]])

        loss = torch.sum(image_anchor_loss) + torch.sum(texy_anchor_loss)
        # loss = CMPM_loss + CMPC_loss

        return loss


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, bias=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

def compute_arcface_loss(arcface,v_embd,t_embd,labels):
    visual_logits = arcface(v_embd,labels)
    textual_logits = arcface(t_embd,labels)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(visual_logits, labels) + criterion(textual_logits, labels)
    return loss