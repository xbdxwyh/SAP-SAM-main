from .gru import build_gru
from .m_resnet import build_m_resnet
from .resnet import build_resnet
from .bert_encoder_gru import build_bertencoder_gru
from .bert import build_bert
from .multi_level_gru import build_multilevel_gru
from .ACmix_ResNet import build_ACmix_ResNet


def build_visual_model(cfg):
    if cfg.MODEL.VISUAL_MODEL in ["resnet50", "resnet101"]:
        return build_resnet(cfg)
    if cfg.MODEL.VISUAL_MODEL in ["m_resnet50","m_multilevel_resnet50", "m_resnet101"]:
        return build_m_resnet(cfg)
    if cfg.MODEL.VISUAL_MODEL in ["acmix_resnet50"]:
        return build_ACmix_ResNet()
    raise NotImplementedError


def build_textual_model(cfg):
    if cfg.MODEL.TEXTUAL_MODEL == "bigru":
        return build_gru(cfg, bidirectional=True)
    elif cfg.MODEL.TEXTUAL_MODEL == "bert_bigru":
        return build_gru(cfg, bidirectional=True, bert_encoder=True)#build_bertencoder_gru(cfg, bidirectional=True)
    elif cfg.MODEL.TEXTUAL_MODEL == "bert_only":
        return build_bert(cfg, bidirectional=False)#build_bertencoder_gru(cfg, bidirectional=True)
    elif cfg.MODEL.TEXTUAL_MODEL == "m_bigru":
        return build_multilevel_gru(cfg, bidirectional=True)
    raise NotImplementedError
