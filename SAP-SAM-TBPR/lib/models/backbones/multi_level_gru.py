import torch
import torch.nn as nn
from transformers import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIPEncoder

from lib.utils.directory import load_vocab_dict



def conv1x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,3), stride=stride,
                     padding=(0,1), groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MultiLevelGRU(nn.Module):
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
        local_part,
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
        
        # config = CLIPTextConfig.from_pretrained(r"config.json")
        # self.lm_encoder = CLIPEncoder(config)
        # self.lm_encoder.load_state_dict(torch.load("clip_encoder_weights.pth"))
        
        self.gru = nn.GRU(
            embed_size,
            hidden_dim,
            num_layers=num_layers,
            dropout=drop_out,
            bidirectional=bidirectional,
            bias=False,
        )

        # self.cnn_local_branch = nn.ModuleList([
        #     Bottleneck(
        #     inplanes=hidden_dim*2, 
        #     planes=hidden_dim*4, 
        #     width=hidden_dim, 
        #     downsample=
        #         nn.Sequential(
        #             conv1x1(hidden_dim*2, hidden_dim*4),
        #             nn.BatchNorm2d(hidden_dim*4),
        #         )
        #     ) for i in range(local_part)]
        # )

        self.local_part = local_part
        self.out_channels = hidden_dim * 2 if bidirectional else hidden_dim
        self.out_linear = nn.Linear(hidden_dim*4,self.out_channels)

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self._init_weight()
        

    # def forward(self, captions):
    #     text = torch.stack([caption.text for caption in captions], dim=1)
    #     text_length = torch.stack([caption.length for caption in captions], dim=1)
    def forward(self, text,text_mask,text_length):
        text_length = text_length.view(-1)
        text = text.view(-1, text.size(-1))  # b x l

        if not self.use_onehot == "yes":
            bs, length = text.shape[0], text.shape[-1]
            text = text.view(-1)  # bl
            text = self.vocab_dict[text].reshape(bs, length, -1)  # b x l x vocab_size
        if self.embed is not None:
            text = self.embed(text)

        # max_length = text.shape[1]
        # mask = [torch.cat([torch.ones((text_length[i])),torch.zeros((max_length-text_length[i]))],dim=0) for i in range(text.shape[0])]
        # mask = torch.stack(mask)

        # bsz, src_len = mask.size()
        # dtype = text.dtype
        # tgt_len = None
        # tgt_len = tgt_len if tgt_len is not None else src_len

        # expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        # inverted_mask = 1.0 - expanded_mask
        # inverted_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

        # text = self.lm_encoder(inputs_embeds = text,attention_mask = inverted_mask.to(text.device))[0]

        gru_out = self.gru_out(text, text_length)
        low_level_global_features, _ = torch.max(gru_out, dim=1)

        #high_level_local_feature = [self.max_pool(self.cnn_local_branch[i](gru_out.unsqueeze(1).permute(0, 3, 1, 2))) for i in range(self.local_part)]
        
        # high_level_global_feature = self.max_pool(torch.cat(high_level_local_feature,dim=2)).squeeze(dim=-1).squeeze(dim=-1)
        # high_level_global_feature = self.out_linear(high_level_global_feature)

        high_level_local_feature = gru_out # [i.squeeze(dim=-1).squeeze(dim=-1) for i in high_level_local_feature]
        
        return _,low_level_global_features,high_level_local_feature


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
        # for m in self.cnn_local_branch.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        for m in self.gru.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, 1)
                nn.init.constant(m.bias.data, 0)


def build_multilevel_gru(cfg, bidirectional):
    use_onehot = cfg.MODEL.GRU.ONEHOT
    hidden_dim = cfg.MODEL.GRU.NUM_UNITS
    vocab_size = cfg.MODEL.GRU.VOCABULARY_SIZE
    embed_size = cfg.MODEL.GRU.EMBEDDING_SIZE
    num_layer = cfg.MODEL.GRU.NUM_LAYER
    drop_out = 1 - cfg.MODEL.GRU.DROPOUT_KEEP_PROB
    root = cfg.ROOT
    local_part = cfg.MODEL.LOCAL_PART

    model = MultiLevelGRU(
        hidden_dim,
        vocab_size,
        embed_size,
        num_layer,
        drop_out,
        bidirectional,
        use_onehot,
        root,
        local_part
    )

    if cfg.MODEL.FREEZE:
        for m in [model.embed, model.gru]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    return model