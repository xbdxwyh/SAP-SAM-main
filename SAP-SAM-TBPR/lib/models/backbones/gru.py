import torch
import torch.nn as nn

from lib.utils.directory import load_vocab_dict
from transformers import BertModel

class GRU(nn.Module):
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
        bert_path = None,
        bert_frozen = False,
        num_bert_layers = 6,
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
            print(vocab_size,vocab_dict.shape[1])
            assert vocab_size == vocab_dict.shape[1]
            self.vocab_dict = torch.tensor(vocab_dict).cuda().float()

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

        if bert_path is not None:
            self.use_bert_encoder = True
            self.bert_frozen = bert_frozen

            self.bert_encoder = BertModel.from_pretrained(bert_path).encoder.layer[0:num_bert_layers]#.eval()
            self.bert_encoder = self.bert_encoder if self.bert_frozen else self.bert_encoder.eval()
            if self.bert_frozen:
                for param in self.bert_encoder.parameters():
                    param.requires_grad = False

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

        gru_out = self.gru_out(text, text_length)
        gru_out, _ = torch.max(gru_out, dim=1)
        return gru_out

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
            if self.bert_frozen:
                with torch.no_grad():
                    for layer in self.bert_encoder:
                        text = layer(hidden_states = text,attention_mask=captions_mask[:,None,None,:].to(text.device))[0]
            else:
                for layer in self.bert_encoder:
                    text = layer(hidden_states = text,attention_mask=captions_mask[:,None,None,:].to(text.device))[0]

        gru_out = self.gru_out(text, text_length)
        gru_out, _ = torch.max(gru_out, dim=1)
        return gru_out

    def gru_out(self, embed, text_length):
        device = embed.device

        _, idx_sort = torch.sort(text_length.cpu(), dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        embed_sort = embed.cpu().index_select(0, idx_sort)
        length_list = text_length[idx_sort]
        pack = nn.utils.rnn.pack_padded_sequence(
            embed_sort, length_list.cpu(), batch_first=True
        )

        gru_sort_out, _ = self.gru(pack.to(device))
        gru_sort_out = nn.utils.rnn.pad_packed_sequence(gru_sort_out.cpu(), batch_first=True)
        gru_sort_out = gru_sort_out[0]

        gru_out = gru_sort_out.index_select(0, idx_unsort)
        return gru_out.to(device)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, 1)
                nn.init.constant(m.bias.data, 0)


def build_gru(cfg, bidirectional, bert_encoder):
    use_onehot = cfg.MODEL.GRU.ONEHOT
    hidden_dim = cfg.MODEL.GRU.NUM_UNITS
    vocab_size = cfg.MODEL.GRU.VOCABULARY_SIZE
    embed_size = cfg.MODEL.GRU.EMBEDDING_SIZE
    num_layer = cfg.MODEL.GRU.NUM_LAYER
    drop_out = 1 - cfg.MODEL.GRU.DROPOUT_KEEP_PROB
    root = cfg.ROOT
    bert_path = cfg.BERT_PATH if bert_encoder else None
    bert_frozen  = cfg.MODEL.FREEZE_BERT
    num_bert_layers = cfg.NUM_BERT_LAYERS

    model = GRU(
        hidden_dim,
        vocab_size,
        embed_size,
        num_layer,
        drop_out,
        bidirectional,
        use_onehot,
        root,
        bert_path,
        bert_frozen,
        num_bert_layers
    )

    if cfg.MODEL.FREEZE:
        for m in [model.embed, model.gru]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    return model
