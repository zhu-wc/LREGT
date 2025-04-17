from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention
from einops.layers.torch import Rearrange

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            MLP(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            MLP(dim, channel_dim, dropout),
        )
    def forward(self, x):
        # print(x.shape)
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs
                                        )
        self.mlp = MixerBlock(dim=512,num_patch=49,token_dim=49,channel_dim=512)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.fc = nn.Linear(2*d_model,d_model)
        self.silu = nn.SiLU()
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))

        local_feature = self.mlp(queries)
        local_feature = self.lnorm2(queries + self.dropout(local_feature))

        fusion_attention = self.silu(self.fc(torch.cat([att,local_feature],dim=-1)))*att

        fusion_attention = self.pwff(fusion_attention)
        return fusion_attention,att,local_feature


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs
                                                  )
                                     for _ in range(N)])
        self.padding_idx = padding_idx

        self.pwff_global = PositionWiseFeedForward(d_model, d_ff, dropout,identity_map_reordering=identity_map_reordering)
        self.pwff_local = PositionWiseFeedForward(d_model, d_ff, dropout,identity_map_reordering=identity_map_reordering)

    def forward(self, input, attention_weights=None):
        attention_mask = (torch.sum(input == 0, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        out = input
        for l in self.layers:
            out,global_out,local_out = l(out, out, out, attention_mask, attention_weights)

        global_out = self.pwff_global(global_out)
        local_out = self.pwff_local(local_out)

        outs = [out.unsqueeze(1),global_out.unsqueeze(1),local_out.unsqueeze(1)]
        outs = torch.cat(outs,1)
        return outs, attention_mask


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None):
        mask = (torch.sum(input, dim=-1) == 0).unsqueeze(-1)
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.masked_fill(mask, 0)  # add by luo
        return super(TransformerEncoder, self).forward(out, attention_weights=attention_weights)
