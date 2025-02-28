import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import make_loss_evaluator

from ...cliora.net.cliora import DioraMLP
from ...cliora.net.utils import ImageEncoder
#from ...cliora.net.trainer import Embed
from ...cliora.blocks.negative_sampler import (
    choose_negative_samples,
    calculate_freq_dist,
    NegativeSampler
)

class Embed(nn.Module):
    def __init__(self, input_size, size):
        super(Embed, self).__init__()
        self.input_size = input_size
        self.size = size
        #self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self.mat1 = nn.Parameter(torch.FloatTensor(size, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def forward(self, x):
        batch_size, length, dim = x.shape
        emb = x.view(-1,dim)
        #emb = self.embeddings(x.view(-1))
        emb_span = torch.mm(emb, self.mat.t()).view(batch_size, length, -1)
        emb_word = torch.mm(emb, self.mat1.t()).view(batch_size, length, -1)
        return emb_span, emb_word


class MoCoHead(nn.Module):
    def __init__(
        self,
        cfg,
        visual_model,
        textual_model,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        self.K = cfg.MODEL.MOCO.K
        self.m = cfg.MODEL.MOCO.M
        self.fc = cfg.MODEL.MOCO.FC
        self.neg_vocab_size = cfg.CLIORA.NEG_VOCAB_SIZE
        self.neg_sample_num = cfg.CLIORA.NEG_SAMPLE_NUM
        self.neg_freq = cfg.CLIORA.NEG_FREQ
        self.v_proj_dim = cfg.CLIORA.PROJ_DIM
        self.t_proj_dim = cfg.CLIORA.PROJ_DIM
        self.cliora_dim = cfg.CLIORA.PROJ_DIM


        self.v_encoder_q = visual_model
        self.t_encoder_q = textual_model
        self.v_encoder_k = copy.deepcopy(visual_model)
        self.t_encoder_k = copy.deepcopy(textual_model)
        for param in self.v_encoder_k.parameters():
            param.requires_grad = False
        for param in self.t_encoder_k.parameters():
            param.requires_grad = False

        if self.fc:
            self.v_fc_q = nn.Sequential(
                nn.Linear(visual_model.out_channels, self.embed_size),
                nn.ReLU(),
                nn.Linear(self.embed_size, self.embed_size),
            )
            self.t_fc_q = nn.Sequential(
                nn.Linear(textual_model.out_channels, self.embed_size),
                nn.ReLU(),
                nn.Linear(self.embed_size, self.embed_size),
            )
            self.v_fc_k = copy.deepcopy(self.v_fc_q)
            self.t_fc_k = copy.deepcopy(self.t_fc_q)
            for param in self.v_fc_k.parameters():
                param.requires_grad = False
            for param in self.t_fc_k.parameters():
                param.requires_grad = False

        self.v_embed_layer = nn.Linear(visual_model.out_channels+self.v_proj_dim*2, self.embed_size)
        #self.v_embed_low_layer = nn.Linear(visual_model.out_channels, self.embed_size)
        # self.v_embed_local_layer = nn.ModuleList(
        #     [nn.Linear(visual_model.out_channels*2, self.embed_size) for i in range(cfg.MODEL.LOCAL_PART)]
        # )

        self.t_embed_layer = nn.Linear(textual_model.out_channels+self.t_proj_dim*2, self.embed_size)
        #self.t_embed_low_layer = nn.Linear(textual_model.out_channels, self.embed_size)
        # self.t_embed_local_layer = nn.ModuleList(
        #     [nn.Linear(textual_model.out_channels*2, self.embed_size) for i in range(cfg.MODEL.LOCAL_PART)]
        # )

        self.v_proj_layer = ImageEncoder(visual_model.out_channels*2,self.v_proj_dim)
        self.t_proj_layer = Embed(textual_model.out_channels,self.t_proj_dim)
        self.diora = DioraMLP(self.cliora_dim, outside=True, normalize="unit", compress=False, share=False)

        self.register_buffer("t_queue", torch.rand(self.embed_size, self.K))
        self.t_queue = F.normalize(self.t_queue, dim=0)
        self.register_buffer("v_queue", torch.rand(self.embed_size, self.K))
        self.v_queue = F.normalize(self.v_queue, dim=0)
        # initialize id label as -1
        self.register_buffer("id_queue", -torch.ones((1, self.K), dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.v_encoder_q.parameters(), self.v_encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(
            self.t_encoder_q.parameters(), self.t_encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        if self.fc:
            for param_q, param_k in zip(
                self.v_fc_q.parameters(), self.v_fc_k.parameters()
            ):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
            for param_q, param_k in zip(
                self.t_fc_q.parameters(), self.t_fc_k.parameters()
            ):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def get_text_embedding(self,captions,text_mask,caption_length):
        _,t_embed,t_embd_local = self.t_encoder_q(captions,text_mask,caption_length)
        _,_,_,_,v_embd_local = self.v_encoder_q(torch.zeros((captions.shape[0],3,384,128)).cuda())
        
        features_span, features_word = self.v_proj_layer(torch.stack(v_embd_local,dim=1))
        emb_span, emb_word = self.t_proj_layer(t_embd_local)

        self.diora(emb_span, emb_word, features_span, features_word)

        inside_h_global = self.diora.chart.inside_h[:,0]
        outside_h_global = self.diora.chart.outside_h[:,0]

        t_embed = torch.cat([t_embed,inside_h_global,outside_h_global],dim=1)

        t_embed = self.t_embed_layer(t_embed)
        return t_embed

    @torch.no_grad()
    def get_visual_embedding(self, images):
        _,_,_,v_embed,v_embd_local = self.v_encoder_q(images)
        captions = torch.zeros((images.shape[0],18),dtype=torch.long).cuda()
        text_mask = torch.zeros((images.shape[0],18),dtype=torch.long).cuda()
        caption_length = torch.ones((images.shape[0],1),dtype=torch.long).cuda()

        _,_,t_embd_local = self.t_encoder_q(captions,text_mask,caption_length)

        features_span, features_word = self.v_proj_layer(torch.stack(v_embd_local,dim=1))
        emb_span, emb_word = self.t_proj_layer(t_embd_local)

        self.diora(emb_span, emb_word, features_span, features_word)

        inside_h_global = self.diora.chart.inside_h[:,0]
        outside_h_global = self.diora.chart.outside_h[:,0]

        v_embed = torch.cat([v_embed,inside_h_global,outside_h_global],dim=1)
        v_embed = self.v_embed_layer(v_embed)
        return v_embed

    @torch.no_grad()
    def _dequeue_and_enqueue(self, v_keys, t_keys, id_keys):
        batch_size = v_keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.v_queue[:, ptr : ptr + batch_size] = v_keys.T
        self.t_queue[:, ptr : ptr + batch_size] = t_keys.T
        self.id_queue[:, ptr : ptr + batch_size] = id_keys.T

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, images, captions, caption_length, text_mask, label):
        N = images.shape[0]

        _,_,low_level_v_feature,high_level_v_feature_global,high_level_v_feature_local = self.v_encoder_q(images)
        low_level_t_feature,high_level_t_feature_global,high_level_t_feature_local = self.t_encoder_q(captions,text_mask,caption_length)
        
        freq = calculate_freq_dist(captions,self.neg_vocab_size)
        negative_sampler = NegativeSampler(freq,self.neg_freq)
        neg_samples = choose_negative_samples(negative_sampler,self.neg_sample_num)#negative_sampler.sample(100)

        emb_span, emb_word = self.t_proj_layer(high_level_t_feature_local)
        features_span, features_word = self.v_proj_layer(torch.stack(high_level_v_feature_local,dim=1))
        # print(
        #     emb_span.shape,
        #     emb_word.shape,
        #     features_span.shape,
        #     features_word.shape,
        # )
        self.diora(emb_span, emb_word, features_span, features_word)

        inside_h_global = self.diora.chart.inside_h[:,0]
        outside_h_global = self.diora.chart.outside_h[:,0]

        v_embed = torch.cat([high_level_v_feature_global,inside_h_global,outside_h_global],dim=1)
        t_embed = torch.cat([high_level_t_feature_global,inside_h_global,outside_h_global],dim=1)

        if self.training:
            if self.fc:
                v_embed_q = self.v_fc_q(v_embed)
                t_embed_q = self.t_fc_q(t_embed)
                v_embed = self.v_embed_layer(v_embed)
                t_embed = self.t_embed_layer(t_embed)
                v_embed_q = F.normalize(v_embed_q, dim=1)
                t_embed_q = F.normalize(t_embed_q, dim=1)
            else:
                v_embed = self.v_embed_layer(v_embed)
                t_embed = self.t_embed_layer(t_embed)
                v_embed_q = F.normalize(v_embed, dim=1)
                t_embed_q = F.normalize(t_embed, dim=1)
            #id_q = torch.stack([caption.get_field("id") for caption in captions]).long()
            id_q = label

            with torch.no_grad():
                self._momentum_update_key_encoder()

                #v_embed_k = self.v_encoder_k(images)
                _,_,_,v_embed_k,v_embd_k_local = self.v_encoder_k(images)
                _,t_embed_k,t_embd_k_local = self.t_encoder_k(captions,text_mask,caption_length)

                emb_span, emb_word = self.t_proj_layer(t_embd_k_local)
                features_span, features_word = self.v_proj_layer(torch.stack(v_embd_k_local,dim=1))

                self.diora(emb_span, emb_word, features_span, features_word)

                inside_h_global = self.diora.chart.inside_h[:,0]
                outside_h_global = self.diora.chart.outside_h[:,0]

                v_embed_k = torch.cat([v_embed_k,inside_h_global,outside_h_global],dim=1)
                t_embed_k = torch.cat([t_embed_k,inside_h_global,outside_h_global],dim=1)

                if self.fc:
                    v_embed_k = self.v_fc_k(v_embed_k)
                else:
                    v_embed_k = self.v_embed_layer(v_embed_k)
                v_embed_k = F.normalize(v_embed_k, dim=1)
                #t_embed_k = self.t_encoder_k(captions)
                
                if self.fc:
                    t_embed_k = self.t_fc_k(t_embed_k)
                else:
                    t_embed_k = self.t_embed_layer(t_embed_k)
                t_embed_k = F.normalize(t_embed_k, dim=1)

            # regard same instance ids as positive sapmles, we need filter them out
            pos_idx = (
                self.id_queue.expand(N, self.K)
                .eq(id_q.unsqueeze(-1))
                .nonzero(as_tuple=False)[:, 1]
            )
            unique, counts = torch.unique(
                torch.cat([torch.arange(self.K).long().cuda(), pos_idx]),
                return_counts=True,
            )
            neg_idx = unique[counts == 1]

            # v positive logits: Nx1
            v_pos = torch.einsum("nc,nc->n", [v_embed_q, t_embed_k]).unsqueeze(-1)
            # v negative logits: NxK
            t_queue = self.t_queue.clone().detach()
            t_queue = t_queue[:, neg_idx]
            v_neg = torch.einsum("nc,ck->nk", [v_embed_q, t_queue])
            # t positive logits: Nx1
            t_pos = torch.einsum("nc,nc->n", [t_embed_q, v_embed_k]).unsqueeze(-1)
            # t negative logits: NxK
            v_queue = self.v_queue.clone().detach()
            v_queue = v_queue[:, neg_idx]
            t_neg = torch.einsum("nc,ck->nk", [t_embed_q, v_queue])

            # low_level_v_feature = self.v_embed_low_layer(low_level_v_feature)
            # low_level_t_feature = self.t_embed_low_layer(low_level_t_feature)

            # high_level_v_feature_local = [layer(feature) for layer,feature in zip(self.v_embed_local_layer,high_level_v_feature_local)]
            # high_level_t_feature_local = [layer(feature) for layer,feature in zip(self.t_embed_local_layer,high_level_t_feature_local)]

            losses = self.loss_evaluator(
                v_embed, 
                t_embed,
                neg_samples.cuda(),#torch.LongTensor(neg_samples).cuda(),
                captions,
                self.diora,
                low_level_v_feature,
                low_level_t_feature,
                high_level_v_feature_local,
                high_level_t_feature_local, 
                v_pos, 
                v_neg, 
                t_pos, 
                t_neg, 
                id_q
            )
            self._dequeue_and_enqueue(v_embed_k, t_embed_k, id_q)
            return losses

        v_embed = self.v_embed_layer(v_embed)
        t_embed = self.t_embed_layer(t_embed)
        outputs = list()
        outputs.append(v_embed)
        outputs.append(t_embed)
        return outputs


def build_moco_head(cfg, visual_model, textual_model):
    return MoCoHead(cfg, visual_model, textual_model)
