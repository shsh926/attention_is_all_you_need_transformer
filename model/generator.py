
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):                                     # linear+softmax
    # vocab: tgt_vocab词典中的所有字都有一个概率，即输出最大概率即为预测值
    def __init__(self, d_model, vocab):
        # d_model = 512
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)               # dim 沿着最后一个维度进行求log_softmax