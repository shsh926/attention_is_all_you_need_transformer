import os
import torch
import numpy as np
from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable
from parser1 import args
from utils import seq_padding, subsequent_mask


class PrepareData:
    def __init__(self):

        # 加载数据
        self.train_en, self.train_cn = self.load_data(args.train_file)
        self.dev_en, self.dev_cn = self.load_data(args.dev_file)

        # 构建单词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)

        # id化
        self.train_en, self.train_cn = self.wordToID(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.wordToID(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)

        # 划分batch + padding + mask
        self.train_data = self.splitBatch(self.train_en, self.train_cn, args.batch_size)
        self.dev_data = self.splitBatch(self.dev_en, self.dev_cn, args.batch_size)

    def load_data(self, path):
        en = []
        cn = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                line = line.strip().split('\t')
                # print(line[1])                             # line[0]-en,line[1]-cn
                # 添加开头结尾标志符
                en.append(["BOS"] + word_tokenize(line[0].lower()) + ["EOS"])
                cn.append(["BOS"] + word_tokenize(" ".join([w for w in line[1]])) + ["EOS"])

        return en, cn

    def build_dict(self, sentences, max_words=50000):
        word_count = Counter()

        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1                                    # 字典形式统计单词

        ls = word_count.most_common(max_words)                        # 排序前50000个最常出现的单词
        # print(ls)                                                   # en,cn前50000个单词

        total_words = len(ls) + 2                                     # 加上sos和结尾eos，总数5002

        # index: 0, x: ('我', 9598)
        # index: 1, x: ('的', 6211)
        # 重新对前50000个单词进行id标注，+2：UNK+PAD
        # {'我': 2, '的': 3, '了': 4}
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}   # ls的形式：('我', 9598), ('的', 6211)
        word_dict['UNK'] = args.UNK                                   # 0
        word_dict['PAD'] = args.PAD                                   # 1

        # {2: '我', 3: '的', 4: '了'}翻转k，v
        index_dict = {v: k for k, v in word_dict.items()}

        return word_dict, total_words, index_dict                    # {word：id}，num，{id：word}

    def wordToID(self, en, cn, en_dict, cn_dict, sort=True):
        length_en = len(en)                                           # 后面没有用到
        length_cn = len(cn)
        # print("length_en:", length_en)                              # 21033和83，为什么两个
        # print("length_cn:", length_cn)                              # 21033和83，

        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        # print(out_en_ids)                     # 返回en每句话的id-[[2, 1748, 4, 3], [2, 1748, 4, 3]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]
        # print(out_cn_ids)                     # 返回才能每句话的id-[2, 48, 451, 7, 4, 3], [2, 634, 127, 635, 7, 4, 3]

        # sort sentences by english lengths
        def len_argsort(seq):

            # print(range(len(seq)))                                # 返回(0,21033),(0,83)
            # x 应该是索引，按照关键字key的len排序？？？
            # seq[x]代表第几句话，是平行语料id值，len(seq[x])返回src，tgt，句子长度：4,6
            # print(sorted(range(len(seq)), key=lambda x: len(seq[x])))   # seq:en每句话的id-[[2, 1748, 4, 3], [2, 1748, 4, 3]]
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))     # range(0,len(seq))，排序句子从短到长？
            # 上述代码排序，按照key表达式操作
            # len(seq):一共多少个句子

        # 把中文和英文按照同样的顺序排序  ???
        if sort:
            sorted_index = len_argsort(out_en_ids)
            # print(len(out_en_ids))
            # print(sorted_index)
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            # print("len(out_en_ids:", len(out_cn_ids))                    # 21033,83,这两个值分别代表什么意思？？？
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]
            # print("len(out_cn_ids:", len(out_cn_ids))                    # 21033,82不应该是训练集个测试集？？？

        return out_en_ids, out_cn_ids

    def splitBatch(self, en, cn, batch_size, shuffle=True):
        # print(len(en))                                                 # len(en)返回：21033，83
        idx_list = np.arange(0, len(en), batch_size)                     # 按照batch_size间隔取值,batch_size=64
        # print(idx_list)
        if shuffle:
            np.random.shuffle(idx_list)                                  # np.random，随机，打乱
        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))      # 几个意思取连续值？？
            # print(batch_indexs)                                  # batch_indexs，array([19968, 19969, 19970, 19971])若干个

        batches = []
        for batch_index in batch_indexs:                        # batch_index为取出一个序列[, , , ]
            batch_en = [en[index] for index in batch_index]     # index每一个值
            batch_cn = [cn[index] for index in batch_index]
            # print(len(batch_en))                                # 64
            # print(len(batch_cn))                                # 64

            batch_cn = seq_padding(batch_cn)                    # 填充padding=0
            # print(batch_cn[12])
            batch_en = seq_padding(batch_en)
            batches.append(Batch(batch_en, batch_cn))           # 平行语料Batch类
            # print(batches)

        return batches
    # def __str__(self):
    #     print(PrepareData())

# PrepareData()

# 构建batch和mask开始
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):

        src = torch.from_numpy(src).to(args.device).long()
        trg = torch.from_numpy(trg).to(args.device).long()

        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)             # 选取句子
        if trg is not None:
            self.trg = trg[:, :-1]                             # 除了最后一个不取，其余纵轴都取，用[src+trg]来预测trg_y
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        # 上面：tgt_mask为经过pad后的tensor,subsequent_mask()得到的结果是布尔值，等价于下三角矩阵
        return tgt_mask                                            # 掩码情况

PrepareData()

