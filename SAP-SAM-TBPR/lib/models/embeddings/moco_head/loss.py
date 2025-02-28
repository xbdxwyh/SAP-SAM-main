import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from transformers import BertModel

from torch import Tensor
import lib.models.losses as losses


class ReconstructionSoftmaxLoss(nn.Module):
    name = 'reconstruct_softmax_loss'

    def __init__(self, embeddings, input_size, size, margin=1, k_neg=3, cuda=False):
        super(ReconstructionSoftmaxLoss, self).__init__()
        self.k_neg = k_neg
        self.margin = margin
        self.input_size = input_size

        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        # self.mat_vis = nn.Parameter(torch.FloatTensor(size, input_size))
        self.lossfn = nn.CrossEntropyLoss()
        self._cuda = cuda
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def forward(self, sentences, neg_samples, diora, info):
        batch_size, length = sentences.shape
        # input_size = self.input_size
        # size = cliora.outside_h.shape[-1]
        k = self.k_neg

        emb_pos = self.embeddings(sentences)
        emb_neg = self.embeddings(neg_samples.unsqueeze(0))

        # Calculate scores.

        cell = diora.outside_h[:, :length].view(batch_size, length, 1, -1)
        #torch.backends.cudnn.enabled = False
        proj_pos = torch.matmul(emb_pos, torch.t(self.mat))
        proj_neg = torch.matmul(emb_neg, torch.t(self.mat))
        # cell = cliora.outside_vh[:, :length].view(batch_size, length, 1, -1)
        # proj_pos = torch.matmul(emb_pos, torch.t(self.mat_vis))
        # proj_neg = torch.matmul(emb_neg, torch.t(self.mat_vis))

        ## The score.
        xp = torch.einsum('abc,abxc->abx', proj_pos, cell)
        xn = torch.einsum('zec,abxc->abe', proj_neg, cell)
        score = torch.cat([xp, xn], 2)

        # Calculate loss.
        inputs = score.view(batch_size * length, k + 1)
        device = torch.cuda.current_device() if self._cuda else None
        outputs = torch.full((inputs.shape[0],), 0, dtype=torch.int64, device=device)

        loss = self.lossfn(inputs, outputs)

        ret = dict(reconstruction_softmax_loss=loss)

        return loss, ret


class ContrastiveLoss(torch.nn.Module):
    name = 'contrastive_loss'

    def __init__(self, margin=1.0, alpha_contr=0.01, use_contr_ce=False):
        super(ContrastiveLoss, self).__init__()
        self.min_val = 1e-8
        self.margin = margin
        self.alpha_contr = alpha_contr
        self.use_contr_ce = use_contr_ce

    def forward(self, batch, diora):
        bs, seq_len = batch.shape
        inside_scores = diora.inside_s.squeeze(-1)  # bs*span_length*1
        outside_scores = diora.outside_s.squeeze(-1)
        # inside_scores = cliora.chart.inside_vs.squeeze(-1)  # bs*span_length*1
        # outside_scores = cliora.chart.outside_vs.squeeze(-1)
        span_length = inside_scores.shape[1]
        device = inside_scores.device

        # Flickr
        all_atten_score = diora.all_atten_score
        assert all_atten_score is not None
        scores = all_atten_score.max(-1).values # bs*bs*span_len
        # TODO: COCO
        # scores = cliora.all_atten_score # bs*bs*span_len

        scores = scores.permute(2, 0, 1)  # span_len*bs*bs
        diagonal = torch.diagonal(scores, 0, -1).unsqueeze(-1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.transpose(1, 2).expand_as(scores) # span_len*bs*bs

        loss_txt = (self.margin + scores - d1).clamp(min=self.min_val)
        loss_img = (self.margin + scores - d2).clamp(min=self.min_val)
        I = (torch.eye(bs) > 0.5).unsqueeze(0).expand_as(scores).to(device)
        loss_txt = loss_txt.masked_fill_(I, 0)
        loss_img = loss_img.masked_fill_(I, 0)

        loss_txt = loss_txt.mean(2) # span_len*bs
        loss_img = loss_img.mean(1) # span_len*bs

        vl_loss = (loss_txt + loss_img).t() # bs*span_len

        span_margs = torch.exp(inside_scores + outside_scores - inside_scores[:, [-1]]) # bs*span_length
        loss_mat = span_margs * vl_loss
        loss = loss_mat[:, :(span_length//2)].sum(-1).mean() * self.alpha_contr
        ret = dict(contrastive_loss=loss)

        return loss, ret


class VGLoss(torch.nn.Module):
    name = 'vg_loss'

    def __init__(self, alpha_vg=0.1):
        super(VGLoss, self).__init__()
        self.min_val = 1e-8
        self.alpha_vg = alpha_vg

    def forward(self, batch, vg_atten_score):
        # bs, seq_len = batch.shape

        batch_size, _, seq_len, _ = vg_atten_score.size()

        # [B, B, seq_len]
        phrase_region_max = vg_atten_score.max(-1).values

        """ V1 """
        phrase_region_scores = phrase_region_max.sum(-1)
        logits = phrase_region_scores.div(
            torch.tensor(seq_len, device=phrase_region_scores.device).expand(batch_size).unsqueeze(1).expand(
                phrase_region_scores.size())
        )

        """ V2 """
        # phrase_region_scores = phrase_region_max.sum(-1)
        # mask = phrase_region_scores.ge(0)
        # logits = phrase_region_scores.div(mask.sum(-1) + 1e-8)

        """ V3 """
        # mask = torch.softmax(phrase_region_max, -1)
        # phrase_region_scores = (phrase_region_max * mask).sum(-1)
        # logits = phrase_region_scores

        targets = torch.arange(
            batch_size, device=phrase_region_scores.device
        )


        loss = self.alpha_vg * F.cross_entropy(logits, targets)
        ret = dict(vg_loss=loss)
        return loss, ret



class LossComputation(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.projection = Parameter(
            torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES),
            requires_grad=True,
        )
        self.epsilon = cfg.MODEL.EMBEDDING.EPSILON
        self.ranking_margin = cfg.RANKING_MARGIN
        # self.T = Parameter(torch.tensor(0.07), requires_grad=True)
        self.T = 0.07
        #self.arcface = losses.ArcFace(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES)
        self.loss_type = cfg.LOSS_TYPE
        self.local_part = cfg.MODEL.LOCAL_PART

        self.vg_loss = VGLoss(alpha_vg= cfg.CLIORA.ALPHA_VG)
        self.reconstruction_loss = ReconstructionSoftmaxLoss(
            BertModel.from_pretrained(cfg.BERT_PATH).embeddings.word_embeddings,
            cfg.CLIORA.PROJ_DIM,
            cfg.CLIORA.PROJ_DIM,
            cfg.CLIORA.MARGIN_RE,
            cfg.CLIORA.NEG_SAMPLE_NUM,
            True
        ).eval()
        self.contrastive_loss = ContrastiveLoss(
            cfg.CLIORA.MARGIN_C,cfg.CLIORA.ALPHA_C
        )

        nn.init.xavier_uniform_(self.projection.data, gain=1)

    def forward(
            self, 
            v_embed:Tensor, 
            t_embed:Tensor, 
            neg_samples,
            captions,
            diora,
            low_v_embd,
            low_t_embd,
            local_v_embd_list,
            local_t_embd_list,
            v_pos:Tensor, 
            v_neg:Tensor, 
            t_pos:Tensor, 
            t_neg:Tensor, 
            labels:Tensor
        ):
        cosine_similarity = nn.CosineSimilarity(-1)

        sim_matrix = cosine_similarity(v_embed.unsqueeze(0),t_embed.unsqueeze(1))

        loss = {}
        for name in self.loss_type:
            if name == "instance_loss":
                loss[name] = losses.instance_loss(
                    self.projection,
                    v_embed,
                    t_embed,
                    labels.reshape(-1),
                    epsilon=self.epsilon,
                )
            elif name == "infonce_loss":
                loss[name] = losses.infonce_loss(
                    v_pos,
                    v_neg,
                    t_pos,
                    t_neg,
                    self.T,
                )
            elif name == "global_align_loss":
                loss[name] = losses.global_align_loss(v_embed, t_embed, labels)
            elif name == "low_align_loss":
                loss[name] = losses.global_align_loss(low_v_embd, low_t_embd, labels)
            elif name == "local_align_loss":
                loss[name] = torch.stack([losses.global_align_loss(local_v_embd_list[i], local_t_embd_list[i], labels) for i in range(self.local_part)]).mean()
            elif name == "global_ranking_loss":
                loss[name] = losses.compute_ranking_loss(sim_matrix, labels, self.ranking_margin)
            elif name == "global_arcface_loss":
                loss[name] = losses.compute_arcface_loss(self.arcface,v_embed,t_embed,labels.reshape(-1))
            elif name == "global_cmpm_loss":
                loss[name] = losses.cmpc_loss(self.projection,v_embed,t_embed,labels.reshape(-1))
            elif name == "cliora_vg_loss":
                loss[name] = self.vg_loss(captions,diora.vg_atten_score)[0] #losses.cmpc_loss(self.projection,v_embed,t_embed,labels.reshape(-1))
            elif name == "cliora_contras_loss":
                loss[name] = self.contrastive_loss(captions,diora)[0]
            elif name == "cliora_recons_loss":
                #print(captions.shape,neg_samples.shape,diora.outside_h.shape)
                loss[name] = self.reconstruction_loss(captions,neg_samples,diora,{})[0]
            else:
                raise NotImplementedError

        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
