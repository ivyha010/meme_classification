import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from compact_bilinear_pooling import CompactBilinearPooling

# ENCODER: CLONE
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# ENCODER
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# ATTENTION
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn   # means x and self_attention in MultiHeadedAttention # 2) Apply attention on all the projected vectors in batch.


# MULTI-HEAD  ATTENTION
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# LAYER NORM
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# SUBLAYER CONNECTION (Residual connection)
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# POSSITION FEED-FORWARD
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        #self.w_1 = nn.Linear(d_model, int(d_model/2))
        #self.w_2 = nn.Linear(int(d_model/2), d_model)
        self.w = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.relu(self.w(x))) # self.w_2(self.dropout(F.relu(self.w_1(x)))) #

# ENCODER LAYER
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# INPUT: EMBEDDING AND SOFTMAX
class Embeddings(nn.Module):  # source
    def __init__(self, numdims, d_model):  # numdims is number of dimensions of scr
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(numdims, d_model)
        self.d_model = d_model
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.float()
        return self.lut(x) * math.sqrt(self.d_model)

#  BASE: ENCODER and a FULLY CONNECTED LAYER
class Encoder_FullyConnected(nn.Module):
    def __init__(self, inputdim1, inputdim2, inputdim3, src_embed, encoder, d_model= 8, dropout=0.1):
        super(Encoder_FullyConnected, self).__init__()
        self.linear1 = nn.Linear(inputdim1, d_model)
        self.linear2 = nn.Linear(inputdim2 + inputdim3, d_model)
        self.src_embed = src_embed
        self.encoder = encoder
        self.cbp_pooling = CompactBilinearPooling(d_model, d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, 1)


    def forward(self, src1, src2, src3):
        reduced1 = self.linear1(src1)
        reduced2 = self.linear2(torch.cat((src2, src3), dim=1))
        src = torch.cat((reduced1.unsqueeze(1), reduced2.unsqueeze(1)), dim=1)
        out_en = self.encoder(src)  # shape: batchsize x 2 x d_model
        out_pool = self.cbp_pooling(out_en[:, 0, :], out_en[:, 1, :])  # shape: batchsize x d_model x 1
        out =  torch.sigmoid(self.linear(self.drop(out_pool)))
        return out.squeeze(1)


# FULL MODEL
def network(input_dim_1, input_dim_2, input_dim_3, N=6, d_model=32, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, dropout)
    embedding = Embeddings(d_model, d_model)
    model = Encoder_FullyConnected(input_dim_1, input_dim_2, input_dim_3, nn.Sequential(embedding), Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), d_model, dropout)
    return model

