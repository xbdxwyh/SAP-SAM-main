import torch
import torch.nn as nn

from transformers import BertModel


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

class BERTWithCNN(nn.Module):
    def __init__(
        self,
        bert_path
    ):
        pass
        super().__init__()
        self.bert_model = BertModel.from_pretrained(bert_path)
        self.bert_model = self.bert_model.eval()
        for m in [self.bert_model]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        
        self.conv1 = conv1x1(self.bert_model.config.hidden_size, self.bert_model.config.hidden_size)
        self.bn1 = nn.BatchNorm2d(self.bert_model.config.hidden_size)
        self.relu = nn.ReLU(inplace=True)

        downsample = nn.Sequential(
            conv1x1(self.bert_model.config.hidden_size, self.bert_model.config.hidden_size),
            nn.BatchNorm2d(self.bert_model.config.hidden_size),
        )

        self.branch1 = nn.Sequential(
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2, 
                       downsample=downsample
                       ),
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2
                       ),
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2
                       )
        )

        self.branch2 = nn.Sequential(
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2, 
                       downsample=downsample
                       ),
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2
                       ),
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2
                       )
        )

        self.branch3 = nn.Sequential(
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2, 
                       downsample=downsample
                       ),
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2
                       ),
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2
                       )
        )

        self.branch4 = nn.Sequential(
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2, 
                       downsample=downsample
                       ),
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2
                       ),
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2
                       )
        )

        self.branch5 = nn.Sequential(
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2, 
                       downsample=downsample
                       ),
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2
                       ),
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2
                       )
        )

        self.branch6 = nn.Sequential(
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2, 
                       downsample=downsample
                       ),
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2
                       ),
            Bottleneck(inplanes=self.bert_model.config.hidden_size, 
                       planes=self.bert_model.config.hidden_size, 
                       width=self.bert_model.config.hidden_size // 2
                       )
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.config = self.bert_model.config

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
    
    def forward(self,captions,mask):
        with torch.no_grad():
            txt = self.bert_model(captions, attention_mask=mask)
            txt = txt[0]
            txt = txt.unsqueeze(1)
            txt = txt.permute(0, 3, 1, 2)
        
        x1 = self.conv1(txt)  # 1024 1 64
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x21 = self.branch1(x1)
        x22 = self.branch2(x1)
        x23 = self.branch3(x1)
        x24 = self.branch4(x1)
        x25 = self.branch5(x1)
        x26 = self.branch6(x1)

        #txt_f3 = self.max_pool(x1).squeeze(dim=-1).squeeze(dim=-1)
        txt_f41 = self.max_pool(x21)
        txt_f42 = self.max_pool(x22)
        txt_f43 = self.max_pool(x23)
        txt_f44 = self.max_pool(x24)
        txt_f45 = self.max_pool(x25)
        txt_f46 = self.max_pool(x26)

        txt_f4 = self.max_pool(torch.cat([txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46], dim=2)).squeeze(dim=-1).squeeze(dim=-1)
        # txt_f41 = txt_f41.squeeze(dim=-1).squeeze(dim=-1)
        # txt_f42 = txt_f42.squeeze(dim=-1).squeeze(dim=-1)
        # txt_f43 = txt_f43.squeeze(dim=-1).squeeze(dim=-1)
        # txt_f44 = txt_f44.squeeze(dim=-1).squeeze(dim=-1)
        # txt_f45 = txt_f45.squeeze(dim=-1).squeeze(dim=-1)
        # txt_f46 = txt_f46.squeeze(dim=-1).squeeze(dim=-1)

        #x1 = self.max_pool(x1).squeeze(dim=-1).squeeze(dim=-1)

        return txt_f4


def build_bert_cnn_textual_model(cfg):
    bert_path = cfg.BERT_PATH
    return BERTWithCNN(bert_path)


class BERTwithGRUModel(nn.Module):
    def __init__(
        self,
        hidden_dim,
        embed_size,
        num_layers,
        drop_out,
        bidirectional,
        bert_path
    ):
        pass
        super().__init__()

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

        self.bert_model = BertModel.from_pretrained(bert_path)
        self.bert_model = self.bert_model.eval()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, 1)
                nn.init.constant(m.bias.data, 0)

    def forward(self, captions, caption_length, text_mask):
        
        with torch.no_grad():
            last_hidden_state = self.bert_model(
                input_ids = captions,
                attention_mask = text_mask
            ).last_hidden_state

        gru_out = self.gru_out(last_hidden_state, caption_length)
        gru_out, _ = torch.max(gru_out, dim=1)
        return gru_out

    def gru_out(self, embed, text_length):
        device = embed.device

        _, idx_sort = torch.sort(text_length, dim=0, descending=True)
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


def build_bert_gru_textual_model(cfg, bidirectional=True):
    hidden_dim = cfg.MODEL.GRU.NUM_UNITS
    embed_size = cfg.MODEL.GRU.EMBEDDING_SIZE
    num_layer = cfg.MODEL.GRU.NUM_LAYER
    drop_out = 1 - cfg.MODEL.GRU.DROPOUT_KEEP_PROB
    bert_path = cfg.BERT_PATH

    model = BERTwithGRUModel(
        hidden_dim,
        embed_size,
        num_layer,
        drop_out,
        bidirectional,
        bert_path
    )

    if cfg.MODEL.FREEZE:
        for m in [model.gru]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    
    if cfg.MODEL.FREEZE_BERT:
        for m in [model.bert_model]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    return model
