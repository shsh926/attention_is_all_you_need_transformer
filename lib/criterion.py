import torch
import torch.nn as nn
from torch.autograd import Variable


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing                         # smoothing=0.4
        self.smoothing = smoothing
        self.size = size                                          # target vocab size 目标语言词表大小
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size                              # # 目标语言词表大小
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))          # 原地填充为self.smoothing/(self.size-2)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)      # 选出值为0的坐标位置
        if mask.dim() > 0:                                         # 没有0，也是>0，判断何意
            true_dist.index_fill_(0, mask.squeeze(), 0.0)          # 在第0维，根据索引mask.squeeze(),进行0的填充，（一行）
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))












