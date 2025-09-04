import torch.nn as nn
from config import DefaultConfig
opt = DefaultConfig()


class TransformEncoder(nn.Module):
    def __init__(self, hidden_size, layer_num=3):
        super(TransformEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(hidden_size) for _ in range(layer_num)])

    def forward(self, h, attn_mask=None):
        h1 = h.clone()
        for layer in self.layers:
            h1 = layer(h1, h1, h1, attn_mask)
        return h1


class TransformDecoder(nn.Module):
    def __init__(self, hidden_size, layer_num=3):
        super(TransformDecoder, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(hidden_size) for _ in range(layer_num)])

    def forward(self, h_q, h_k, h_v):
        q, k, v = h_q.clone(), h_k.clone(), h_v.clone()
        for layer in self.layers:
            q = layer(q, k, v)
        return q


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, head_num=8):
        super(TransformerLayer, self).__init__()
        self.MHA = nn.MultiheadAttention(hidden_size, head_num)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.FF = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, q, k, v, attn_mask=None):
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        h = self.norm1((q + self.MHA(q, k, v, attn_mask=attn_mask)[0]))
        h = h.permute(1, 0, 2)
        h = self.norm2((h + self.FF(h)))
        return h
