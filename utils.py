import copy
import torch.nn as nn
import numpy as np
import torch

# 深度copy6次
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 处理测试数据
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)                                                            # src句子长度paddding
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')    # tgt-上三角矩阵
    return torch.from_numpy(subsequent_mask) == 0                          # 弄成下三角矩阵，即mask掉后面的词

# print(subsequent_mask(3))
