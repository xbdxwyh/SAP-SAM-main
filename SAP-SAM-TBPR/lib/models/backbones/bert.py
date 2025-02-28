import torch
import torch.nn as nn

from transformers import BertConfig,BertModel

from lib.utils.directory import load_vocab_dict
from transformers.models.bert.modeling_bert import BertEncoder


class BertLayers(nn.Module):
    def __init__(
        self,
        hidden_dim,
        vocab_size,
        embed_size,
        num_layers,
        drop_out,
        bidirectional,
        use_onehot,
        root,
        bert_path = None
    ):
        super().__init__()

        self.use_onehot = use_onehot

        # word embedding
        if use_onehot == "yes":
            self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        else:
            if vocab_size == embed_size:
                self.embed = None
            else:
                self.embed = nn.Linear(vocab_size, embed_size)

            vocab_dict = load_vocab_dict(root, use_onehot)
            assert vocab_size == vocab_dict.shape[1]
            self.vocab_dict = torch.tensor(vocab_dict).cuda().float()

        #self.bert_encoder = BertEncoder(bert_config)

        self.out_channels = hidden_dim * 2 if bidirectional else hidden_dim

        if bert_path is not None:
            self.use_bert_encoder = True
            self.bert_encoder = BertModel.from_pretrained(bert_path).encoder.layer[0:num_layers]#.eval()
        

    def forward_mmtbpr(self, captions, text_length, captions_mask):
        #text = torch.stack([caption.text for caption in captions], dim=1)
        #text_length = torch.stack([caption.length for caption in captions], dim=1)
        text = captions

        text_length = text_length.view(-1)
        text = text.view(-1, text.size(-1))  # b x l

        if not self.use_onehot == "yes":
            bs, length = text.shape[0], text.shape[-1]
            text = text.view(-1)  # bl
            text = self.vocab_dict[text].reshape(bs, length, -1)  # b x l x vocab_size
        if self.embed is not None:
            text = self.embed(text)
        
        if self.use_bert_encoder:
            #with torch.no_grad():
            for layer in self.bert_encoder:
                text = layer(hidden_states = text,attention_mask=captions_mask[:,None,None,:].to(text.device))[0]

        # gru_out = self.gru_out(text, text_length)
        # gru_out, _ = torch.max(gru_out, dim=1)
        #global_feature = text[:,0]
        global_feature , _ = torch.max(text, dim=1)
        return global_feature



def build_bert(cfg, bidirectional):
    use_onehot = cfg.MODEL.GRU.ONEHOT
    hidden_dim = cfg.MODEL.GRU.NUM_UNITS
    vocab_size = cfg.MODEL.GRU.VOCABULARY_SIZE
    embed_size = cfg.MODEL.GRU.EMBEDDING_SIZE
    num_layer = cfg.MODEL.GRU.NUM_LAYER
    drop_out = 1 - cfg.MODEL.GRU.DROPOUT_KEEP_PROB
    bert_path = cfg.BERT_PATH

    root = cfg.ROOT

    model = BertLayers(
        hidden_dim,
        vocab_size,
        embed_size,
        num_layer,
        drop_out,
        bidirectional,
        use_onehot,
        root,
        bert_path
    )

    if cfg.MODEL.FREEZE:
        for m in [model.embed,model.bert_encoder]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    return model
