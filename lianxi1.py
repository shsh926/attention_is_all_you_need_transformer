import copy

import torch
from torch import nn
import numpy as np
from utils import clones
import math
from torch.autograd import Variable
# max_len = 10
# d_model = 20
# print(max_len *- d_model)

# import nltk
# nltk.download()
# nltk.download('punkt')
# from nltk.book import *
# pe = torch.zeros(max_len, d_model)
# print(pe)
# position = torch.arange(0., max_len).unsqueeze(1)
# print(position)
#
# print(math.log(10000.0))
# a = torch.tensor(10)
# d_model = torch.tensor(10000)
# print(torch.exp(a))
# print(torch.exp(math.log(100.0) / d_model))
# c1 = math.log(100.0)
# c2 = d_model
# c3 = c1/c2
# print(torch.exp(c3))
# max_len = 50
# d_model = 12
# pe = torch.zeros(max_len, d_model)
# position = torch.arange(0., max_len).unsqueeze(1)
# print(pe)
# print(position)
# a = pe[:, 0::2]
# print(a)
# print(torch.ones(a))

# a = torch.tensor(1.0)
# b = a.clone()
# print(b)
# print(id(a))
# print(id(b))
# a = torch.tensor(2)
# assert 1==1
# assert 1==0、
# b = clones(a, 9)
# print(b)
# a = np.arange(10)
# print(a)
# print(a[:-1])
# for elem in [1,2,3]:
#     print(elem)
# a = torch.tensor([[[1.0], [2.0], [3]]])
# b = copy.deepcopy(a)
# print(id(a))
# print(id(b))
# print(a.view(2,1))
# print(a.dim())
# print(a.dim(1))
# class Maoamo(nn.Module):
#     def __init__(self, name, xingbie):
#         super(Maoamo,self).__init__()
#         self.name = name
#         self.xingbie = xingbie
#         print("我的小猫的名字是：", self.name, "性别是：", self.xingbie)
# A = Maoamo("小花花", "壳子")
# # Animal()
# # Animal()

# max_len = 50
# position = torch.arange(0, max_len).unsqueeze(1)
# print(position)


# class PositionalEncoding(nn.Module):
#   "Implement the PE function."
#   def __init__(self, d_model, dropout, max_len=5000):
#     #d_model=512,dropout=0.1,
#     #max_len=5000代表事先准备好长度为5000的序列的位置编码，其实没必要，
#     #一般100或者200足够了。
#     super(PositionalEncoding, self).__init__()
#     self.dropout = nn.Dropout(p=dropout)
#
#     # Compute the positional encodings once in log space.
#     pe = torch.zeros(max_len, d_model)
#     #(5000,512)矩阵，保持每个位置的位置编码，一共5000个位置，
#     #每个位置用一个512维度向量来表示其位置编码
#     position = torch.arange(0, max_len).unsqueeze(1)
#     # (5000) -> (5000,1)
#     div_term = torch.exp(torch.arange(0, d_model, 2) *
#       -(math.log(10000.0) / d_model))
#       # (0,2,…, 4998)一共准备2500个值，供sin, cos调用
#     pe[:, 0::2] = torch.sin(position * div_term) # 偶数下标的位置
#     pe[:, 1::2] = torch.cos(position * div_term) # 奇数下标的位置
#     pe = pe.unsqueeze(0)
#     print(pe.shape)
#     # print(pe[:, :x.size(1)])
#     print(pe)
#     # (5000, 512) -> (1, 5000, 512) 为batch.size留出位置
#     self.register_buffer('pe', pe)
#   def forward(self, x):
#     x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
#     # print("x:", x)
#     # 接受1.Embeddings的词嵌入结果x，
#     #然后把自己的位置编码pe，封装成torch的Variable(不需要梯度)，加上去。
#     #例如，假设x是(30,10,512)的一个tensor，
#     #30是batch.size, 10是该batch的序列长度, 512是每个词的词嵌入向量；
#     #则该行代码的第二项是(1, min(10, 5000), 512)=(1,10,512)，
#     #在具体相加的时候，会扩展(1,10,512)为(30,10,512)，
#     #保证一个batch中的30个序列，都使用（叠加）一样的位置编码。
#     return self.dropout(x) # 增加一次dropout操作
# # 注意，位置编码不会更新，是写死的，所以这个class里面没有可训练的参数。
#
# PositionalEncoding(d_model=512, dropout=0.1)

# a = torch.tensor([1, 0, 2, 3])
# b = a.masked_fill(mask=torch.ByteTensor([1, 1, 0, 0]), value=torch.tensor(-1e9))
# print(a)
# print(b)

# mask = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
# mask = mask.reshape(3, 2)
# # mask = None
# print(mask.shape)
# print("mask:", mask)
# mask = mask.unsqueeze(1)
# print(mask.shape)
# print(mask)
# a = (1, 2, 3)
# b = (4, 5, 6)
# c = zip(a, b)
# print(list(c))

# for l, x in zip((1, 2, 3), (4, 5, 6)):
#   print(l, x)
# features = 512
# a_2 = (torch.ones(features))
# print(a_2)

# x = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
# y = x.reshape(1, 2, 3)
# print(y.shape)
# print(y)
# print(y.size(-1))
# mean = y.mean(-1, keepdim=True)
# # mean = y.mean()
# print(mean.shape)
# print(mean)

# class LayerNorm(nn.Module):
#   "Construct a layernorm module (See citation for details)."
#
#   def __init__(self, features, eps=1e-6):
#     # features=d_model=512, eps=epsilon 用于分母的非0化平滑
#     super(LayerNorm, self).__init__()
#     self.a_2 = nn.Parameter(torch.ones(features))
#     # a_2 是一个可训练参数向量，(512)
#     self.b_2 = nn.Parameter(torch.zeros(features))
#     # b_2 也是一个可训练参数向量, (512)
#     self.eps = eps
#
#   def forward(self, x):
#     # x 的形状为(batch.size, sequence.len, 512)
#     mean = x.mean(-1, keepdim=True)
#     # 对x的最后一个维度，取平均值，得到tensor (batch.size, seq.len)
#     std = x.std(-1, keepdim=True)
#     # 对x的最后一个维度，取标准方差，得(batch.size, seq.len)
#     print(self.a_2 * (x - mean) / (std + self.eps) + self.b_2)
#     return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
#     # 本质上类似于（x-mean)/std，不过这里加入了两个可训练向量
#     # a_2 and b_2，以及分母上增加一个极小值epsilon，用来防止std为0
#     # 的时候的除法溢出
# LayerNorm(512)
# def subsequent_mask(size):
#   "Mask out subsequent positions."
#   # e.g., size=10
#   attn_shape = (1, size, size)  # (1, 10, 10)
#   subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
#
#
# # triu: 负责生成一个三角矩阵，k-th对角线以下都是设置为0
# # 上三角中元素为1.
#   print(torch.from_numpy(subsequent_mask) == 0)
#   return torch.from_numpy(subsequent_mask) == 0
# # 反转上面的triu得到的上三角矩阵，修改为下三角矩阵。
# # subsequent_mask(10)
# size = 10
# attn_shape = (1, size, size)
# print(np.triu(np.ones(attn_shape), k=1).astype('uint8'))
# a = torch.tensor([[[1.0], [2.3], [3.9]]])
# b = a.dim()
# print(a.shape)
# print(b)
# tgt = torch.tensor([[2, 1, 3]])
# tgt = torch.ones(1, 5, 4)
# print(tgt)
# print(tgt.view(-1))
# pad = 0
# print(tgt)
# print(tgt.data)
# tgt_mask = (tgt != pad).unsqueeze(-2)
# print(tgt_mask)
# print(tgt.type_as())
# print(type(tgt))

# import numpy as np
# import matplotlib.pyplot as plt
# np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
#
# def plotPE(dmodel, numw, width=10, height=10):
#     dmodel = 512
#     numw = 200
#     pematrix = np.zeros((numw, dmodel))
#     for pos in range(0, numw): # 20 words
#         for i in range(0, dmodel): # 512-dimension
#             if i % 2 == 0:
#                 p = np.sin(pos/np.power(10000.0, i/dmodel))
#             else:
#                 p = np.cos(pos/np.power(10000.0, (i-1)/dmodel))
#             pematrix[pos][i] = p
#     plt.figure(figsize=(width, height))
#     print(pematrix)
#     plt.imshow(pematrix)
#     plt.show()
#
# plotPE(dmodel=512, numw=200)




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square, you can specify with a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net = Net()
# print(net)
#
# input = torch.randn(1, 1, 32, 32)
# out = net(input)
# print(out)

# pe = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
# pe[:, :x.size(1)]
# pe = pe.view(-1, 2, 3)
# print(pe.view(2, 3))
# print(pe.size(-1))
# print(pe[1])
# c = copy.deepcopy
#
# a = 90
# print(c(a))
# a += 1
# print(a)
# print(c)
# d_model = 512.0
# a = math.sqrt(d_model)
# print(a)
# from collections import Counter
# c = Counter()
# a = 'gallahad'
# for i in a:
#     c[i] += 1
# print(c)

# print(c)
# print(set(c))


# ls = [('我', 9598), ('的', 6211), ('了', 4578)]
# word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
# # print(word_dict)
#
# index_dict = {v: k for k, v in word_dict.items()}
# print(index_dict)
# print(index_dict.get())
# a = 4
# b = [1, 2, 3, 4]
# key = lambda a: a**3
# print(lambda a: a**3)
# print(key)
# print(key(4))
# a = [1, 2, 3, 4, 5, 9, 90]
# print(range(len(a)))
# list = ['a', 'bc', 'defg', 'handsome', 'qwerrtyyuu']
# print(sorted(list, key=lambda x: len(x)))

# ['qwerrtyyuu', 'handsome', 'defg', 'bc', 'a']

# batch_size = 2
# en = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# idx_list = np.arange(0, len(en), batch_size)
# print(idx_list)
# # print(np.arange(1, 5))
# idx = 10
# batch_size = 15
# en = 45
# id = []
# id.append(np.arange(idx, min(idx + batch_size, en)))
# print(id)
# from torch.autograd import Variable
# # tgt_mask = torch.tensor([[[1,1,1,1]],[[1,1,1,0]],[[1,1,1,1]]],dtype=torch.uint8)
# tgt_mask = torch.tensor([[[1,1,1,1]],[[1,1,1,1]]],dtype=torch.uint8)
# # print(tgt_mask)
#
# subsequent_mask = torch.tensor([[[ True, False, False],[ True,  True, False],[ True,  True,  True]]])
#
# subsequent_mask = torch.tensor([[[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]]],dtype=torch.uint8)
# # print(subsequent_mask)
# # print(Variable(subsequent_mask))
# output = (tgt_mask & Variable(subsequent_mask.type_as(tgt_mask)))
# print(output)


# id = [[2, 1748, 4, 3], [2, 1748, 4, 3], [1, 2, 3, 4, 5, 6, 7, 8, 90]]
# print(sorted(range(len(id)), key=lambda x: len(id[x])))
# print(sorted(range(len(id))))
# for i in id:
#     print(i)
#     for j in i:
#         print(j)

# listC = [('e', 4), ('o', 2), ('!', 5), ('v', 3), ('l', 1)]
#由元组构成的列表
# print(sorted(listC, key=lambda x: x[0]))
#[('l', 1), ('o', 2), ('v', 3), ('e', 4), ('!', 5)]
# en = [20,4]
# a = torch.tensor([1,2])
# print(a.size())
# print(a)
# print(len(a))
# true_dist = torch.tensor([[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
#         [-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
#         [-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233]])
#
# true_dist.fill_(1.11)
# print(true_dist)
# padding_idx = 0.0
# target = torch.tensor([[2.0], [1.0], [0.0]])
# # # print(target)
# # # mask = torch.nonzero(target.data == padding_idx)
# # mask = torch.tensor([1,2,3])
# #
# # # mask = torch.nonzero(target.data)
# # # print(mask)
# # # print(mask.dim())
# print(target.data == padding_idx)
# mask = torch.nonzero(target.data == padding_idx)
# print(mask)
# print(mask.dim())

# index = torch.LongTensor([[0],[1],[1],[2]])

# print(index.squeeze().dim())
# print(index.dim())
# tensor([0, 1, 1, 2])

# src = torch.tensor([[2,1,3,4],[2,3,1,4]])
# # print(src)
# pad = 0
# src_mask = (src != pad).unsqueeze(-2)
# print(src_mask)

data= torch.tensor([[[0, 2, 3]], [[4, 5, 6]]])

# print(data.size())
# print("data:", data)
# print(data.size(1))
# print(data.size(2))
# print(data[:, -1])
max_data, indices = torch.max(data, dim=-1)
# print("max_data:", max_data)
# print("indices:", indices)
# print("indices:", indices.data[0])
src_mask = (data != 0)
print(src_mask)
print("你好吗%s" % "一般般趴")


































