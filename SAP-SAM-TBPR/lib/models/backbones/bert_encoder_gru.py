import torch
import torch.nn as nn

from transformers import BertConfig

from lib.utils.directory import load_vocab_dict
from transformers.models.bert.modeling_bert import BertEncoder


class BertEncoderGRU(nn.Module):
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
        num_hidden_layers,
        num_attention_heads
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

        bert_config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
        )

        self.bert_encoder = BertEncoder(bert_config)

        self.gru = nn.GRU(
            embed_size,
            hidden_dim,
            num_layers=num_layers,
            dropout=drop_out,
            bidirectional=bidirectional,
            bias=False,
        )
        self.out_channels = hidden_dim * 2 if bidirectional else hidden_dim

        self._init_weight()

    def forward(self, captions):
        text = torch.stack([caption.text for caption in captions], dim=1)
        text_length = torch.stack([caption.length for caption in captions], dim=1)

        text_length = text_length.view(-1)
        text = text.view(-1, text.size(-1))  # b x l

        if not self.use_onehot == "yes":
            bs, length = text.shape[0], text.shape[-1]
            text = text.view(-1)  # bl
            text = self.vocab_dict[text].reshape(bs, length, -1)  # b x l x vocab_size
        if self.embed is not None:
            text = self.embed(text)

        max_length = text.shape[1]
        mask = [torch.cat([torch.ones((text_length[i])),torch.zeros((max_length-text_length[i]))],dim=0) for i in range(text.shape[0])]
        mask = torch.stack(mask)

        text = self.bert_encoder(hidden_states=text,attention_mask=mask[:,None,None,:].to(text.device))[0]

        gru_out = self.gru_out(text, text_length)
        gru_out, _ = torch.max(gru_out, dim=1)
        return gru_out

    def gru_out(self, embed, text_length):

        _, idx_sort = torch.sort(text_length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        embed_sort = embed.index_select(0, idx_sort)
        length_list = text_length[idx_sort]
        pack = nn.utils.rnn.pack_padded_sequence(
            embed_sort, length_list.cpu(), batch_first=True
        )

        gru_sort_out, _ = self.gru(pack)
        gru_sort_out = nn.utils.rnn.pad_packed_sequence(gru_sort_out, batch_first=True)
        gru_sort_out = gru_sort_out[0]

        gru_out = gru_sort_out.index_select(0, idx_unsort)
        return gru_out


    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.data, 1)
                nn.init.constant(module.bias.data, 0)
            elif isinstance(module, nn.Linear):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            # elif isinstance(module, nn.Embedding):
            #     module.weight.data.normal_(mean=0.0, std=0.02)
            #     if module.padding_idx is not None:
            #         module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


def build_bertencoder_gru(cfg, bidirectional):
    use_onehot = cfg.MODEL.GRU.ONEHOT
    hidden_dim = cfg.MODEL.GRU.NUM_UNITS
    vocab_size = cfg.MODEL.GRU.VOCABULARY_SIZE
    embed_size = cfg.MODEL.GRU.EMBEDDING_SIZE
    num_layer = cfg.MODEL.GRU.NUM_LAYER
    drop_out = 1 - cfg.MODEL.GRU.DROPOUT_KEEP_PROB
    num_hidden_layers = cfg.MODEL.NUM_BERT_LAYERS
    num_attention_heads = cfg.MODEL.NUM_BERT_HEAD

    root = cfg.ROOT

    model = BertEncoderGRU(
        hidden_dim,
        vocab_size,
        embed_size,
        num_layer,
        drop_out,
        bidirectional,
        use_onehot,
        root,
        num_hidden_layers,
        num_attention_heads
    )

    if cfg.MODEL.FREEZE:
        for m in [model.embed,model.bert_encoder,model.gru]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    return model
