import torch
import math, copy
import torch.nn.functional as F
import torch.nn as nn

from utils import clones

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)                                                     # 最后一个维应该用该是512
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)     # 论文中公式

    if mask is not None:                                                     # 在0的位置补最小数，防止后面变换为0
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn                               # p_attn貌似只是为了画图显示，后面没有用到

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0

        self.d_k = d_model // h                                              # 64
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)                # 三个linear用于训练q、k、v
        self.attn = None                                                     # 最后一个用于传入线性层
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)                                             # nbatchs-batch_size

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)                                 # 在传入线性层计算















