from torch import nn

#from .backbones import build_textual_model, build_visual_model
from .backbones import build_visual_model,build_textual_model

#from transformers import BertModel
from .backbones.SwinV2TBPRModel import SwinV2TBPRModel
from .backbones.bert_gru import build_bert_gru_textual_model,build_bert_cnn_textual_model

from .embeddings import build_embed
from .embeddings.moco_head.mmtbpr_cliora_head import build_moco_head


class MMTBPRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        textual_model = build_textual_model(cfg)#BertModel.from_pretrained(cfg.BERT_PATH)
        visual_model = build_visual_model(cfg)#SwinV2TBPRModel.from_pretrained(cfg.SWINV2_PATH)

        if cfg.MODEL.EMBEDDING.EMBED_HEAD == "moco":
            self.embed_model = build_moco_head(
                cfg, visual_model, textual_model
            )
            self.embed_type = "moco"
        else:
            self.visual_model = visual_model
            self.textual_model = textual_model
            self.embed_model = build_embed(
                cfg, self.visual_model.out_channels, self.textual_model.out_channels
            )
            self.embed_type = "normal"

    def forward(self, images, captions, caption_length, mask, label):
        if self.embed_type == "moco":
            return self.embed_model(images, captions, caption_length, mask, label)

        visual_feat = self.visual_model(images)
        textual_feat = self.textual_model(captions)

        outputs_embed, losses_embed = self.embed_model(
            visual_feat, textual_feat, captions
        )

        if self.training:
            losses = {}
            losses.update(losses_embed)
            return losses

        return outputs_embed


def build_mmtbpr_cliora_model(cfg):
    return MMTBPRModel(cfg)
