import torch
import torch.nn as nn

from parser1 import args


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)                                # 在最后轴上求平均
        std = x.std(-1, keepdim=True)                                  # 在最后轴上求方差
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2     # 防止分母为0，layer层归一化


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))               # sublayer是多头或者全连接的输出，残差连接+归一化