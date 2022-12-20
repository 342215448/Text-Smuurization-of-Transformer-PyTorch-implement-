"""
    author:Peinan Feng
    date:2022.10.27
    description:The implement of Text-Summarization in <Code with Paper>
"""
from torchtext import data
import spacy
import re
from tqdm import tqdm
import pandas as pd
import os
import torch
import numpy as np
import math

# 超参数 ——————————————————————————————————————
BatchSize = 50
Device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CheckPoint = 0
D_Model = 512
Heads = 8
Dropout = 0.1
N_Layers = 6
LearningRate = 0.0001
Epochs = 300
PrintEvery = 1

# 定义local-attention超参————————————————————————
Window = 2
N_FULL = 2
N_local = 1
N_cross = 3


# ——————————————————————————————————————————————


# 创建下三角矩阵 return:np_mask
def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.to(Device)
    return np_mask


# 创建mask return:src_mask, trg_mask
def create_masks(src, trg, src_pad, trg_pad):
    src_mask = (src != src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size)
        trg_mask = trg_mask & np_mask.to(Device)

    else:
        trg_mask = None
    return src_mask, trg_mask


# 获取数列长度 return:len
def get_len(train):
    for i, b in enumerate(train):
        pass

    return i


# 定义分词器 使用Spacy分词
class tokenize(object):
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence):
        sentence = re.sub(
            r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


# 读取原始数据 return:raw_src_data, raw_trg_data
def read_data(src_data, trg_data):
    # 用于判断数据路径
    if src_data is not None:
        try:
            raw_src_data = open(src_data).read().strip().split('\n')
        except:
            print("error: '" + src_data + "' file not found")
            quit()
    if trg_data is not None:
        try:
            raw_trg_data = open(trg_data).read().strip().split('\n')
        except:
            print("error: '" + trg_data + "' file not found")
            quit()
    return raw_src_data, raw_trg_data


# 创建Field类 return:(SRC, TRG)
def create_fields(src_lang, trg_lang):
    # set the Spacy dict
    spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it', 'nl']
    # judge the language module
    if src_lang not in spacy_langs:
        print('invalid src language: ' + src_lang + 'supported languages : ' + spacy_langs)
    if trg_lang not in spacy_langs:
        print('invalid trg language: ' + trg_lang + 'supported languages : ' + spacy_langs)

    print("loading spacy tokenizers...")

    t_src = tokenize(src_lang)
    t_trg = tokenize(trg_lang)

    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    return (SRC, TRG)


# 创建数据集 返回train_iter, src_pad, trg_pad, train_len
def create_dataset(raw_src_data, raw_trg_data, SRC, TRG):
    print("creating dataset and iterator... ")
    print("reading raw data...")
    # 原始数据
    raw_data = {'src': [line for line in tqdm(raw_src_data)], 'trg': [line for line in tqdm(raw_trg_data)]}

    # 测试少数据量情况 正常运行请注释掉----------------------------------
    raw_data['src'] = raw_data['src'][:100]
    raw_data['trg'] = raw_data['trg'][:100]
    # --------------------------------------------------------------

    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    # mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    # df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)

    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=BatchSize, device=Device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)

    os.remove('translate_transformer_temp.csv')

    # 源语料字典
    SRC.build_vocab(train)
    # 目标语料字典
    TRG.build_vocab(train)

    src_pad = SRC.vocab.stoi['<pad>']
    trg_pad = TRG.vocab.stoi['<pad>']

    train_len = get_len(train_iter)

    return train_iter, src_pad, trg_pad, train_len


# 为了解决读取数据过慢 实现一个自己的类
class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


# 返回batch_size的最大长度
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


# ——————————————————————————————————————————————— Module ————————————————————————————————————————————————————
import torch.nn as nn
import copy
import math
from torch.autograd import Variable
import torch.nn.functional as F


# 进行Embedding词嵌入层的构造
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model).to(Device)

    def forward(self, x):
        return self.embed(x)


# 进行位置编码层的构造
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False).to(Device)
        x = x + pe
        return self.dropout(x)


# 使用标准正态分布的方法来进行行归一化
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


# ————————————————————————————————————————————————   Local Self-Attention   —————————————————————————————————————————————————————
# 计算attention return:output
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


# 将q、k矩阵处理成local数据形式 （还未处理rate步长）
def converse_qk(q, k):
    q1 = q.size(0)
    q2 = q.size(1)
    q3 = q.size(2)
    q4 = q.size(3)
    k1 = k.size(0)
    k2 = k.size(1)
    k3 = k.size(2)
    k4 = k.size(3)
    list_q = []
    list_k = []
    print("sliding the local window...")
    for i in range(q3):
        if i == 0:
            tem1 = q[:, :, :Window + 1, :].cpu()
            tem2 = torch.zeros(q1, q2, q3 - Window - 1, q4)
            list_q.append(torch.cat([tem1, tem2], 2).to(Device))
            del tem1, tem2
        elif i == 1:
            tem1 = q[:, :, :Window + 2, :].cpu()
            tem2 = torch.zeros(q1, q2, q3 - Window - 2, q4)
            list_q.append(torch.cat([tem1, tem2], 2).to(Device))
            del tem1, tem2
        elif i == q3 - 1:
            tem1 = q[:, :, q3 - Window - 1:, :].cpu()
            tem2 = torch.zeros(q1, q2, q3 - Window - 1, q4)
            list_q.append(torch.cat([tem2, tem1], 2).to(Device))
            del tem1, tem2
        elif i == q3 - 2:
            tem1 = q[:, :, q3 - Window - 2:, :].cpu()
            tem2 = torch.zeros(q1, q2, q3 - Window - 2, q4)
            list_q.append(torch.cat([tem2, tem1], 2).to(Device))
            del tem1, tem2
        else:
            tem2 = q[:, :, i - Window:i + 1 + Window, :].cpu()
            tem1 = torch.zeros(q1, q2, i - Window, q4)
            tem3 = torch.zeros(q1, q2, q3 - 1 - Window - i, q4)
            list_q.append(torch.cat([torch.cat([tem1, tem2], 2), tem3], 2).to(Device))
            del tem1, tem2, tem3
    for i in range(k3):
        if i == 0:
            tem1 = k[:, :, :Window + 1, :].cpu()
            tem2 = torch.zeros(k1, k2, k3 - Window - 1, k4)
            list_k.append(torch.cat([tem1, tem2], 2).to(Device))
            del tem1, tem2
        elif i == 1:
            tem1 = k[:, :, :Window + 2, :].cpu()
            tem2 = torch.zeros(k1, k2, k3 - Window - 2, k4)
            list_k.append(torch.cat([tem1, tem2], 2).to(Device))
            del tem1, tem2
        elif i == k3 - 1:
            tem1 = k[:, :, k3 - Window - 1:, :].cpu()
            tem2 = torch.zeros(k1, k2, k3 - Window - 1, k4)
            list_k.append(torch.cat([tem2, tem1], 2).to(Device))
            del tem1, tem2
        elif i == k3 - 2:
            tem1 = k[:, :, k3 - Window - 2:, :].cpu()
            tem2 = torch.zeros(k1, k2, k3 - Window - 2, k4)
            list_k.append(torch.cat([tem2, tem1], 2).to(Device))
            del tem1, tem2
        else:
            tem2 = k[:, :, i - Window:i + 1 + Window, :].cpu()
            tem1 = torch.zeros(k1, k2, i - Window, k4)
            tem3 = torch.zeros(k1, k2, k3 - 1 - Window - i, k4)
            list_k.append(torch.cat([torch.cat([tem1, tem2], 2), tem3], 2).to(Device))
            del tem1, tem2, tem3

    return list_q, list_k


def mul_torch_matmul(q, k, d_k):
    return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)


# 计算local self-attention 实际上就是将q、k于window外的部分赋0值，然后再加起来形成新的score 因为还是self-attention 因此kqv同源 纬度相同
def local_attention(q_list, k_list, q, k, v, d_k, mask=None, dropout=None):
    seq_len = q.size(-2)
    # 建立新的scores矩阵 全0
    # scores = torch.zeros_like(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)).to(Device)
    scores = torch.zeros_like(mul_torch_matmul(q, k, d_k))
    # local_layers = get_clones(mul_torch_matmul(q, k, d_k), N_Layers)
    print("caculating the score...")
    for i in range(seq_len):
        score_tem = mul_torch_matmul(q_list[i], k_list[i], d_k)
        scores += score_tem
    # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


# 计算cross-attention
def cross_attention(q, k, v, d_k, mask=None, dropout=None):
    fenmu = 0
    output = 0
    LayerNorm = Norm(q.size(-1), 1e-6).to(Device)
    for sj in enumerate(v):
        for sl in enumerate(k):
            fenmu += torch.exp(torch.matmul(q, sl[1].transpose(-2, -1))) * math.sqrt(d_k)
        fenzi = torch.exp(torch.matmul(q, k.transpose(-2, -1)))
        # tem = torch.matmul((fenzi/fenmu),sj[1])
        score = torch.matmul((fenzi / fenmu), sj[1])
        if mask is not None:
            mask = mask.unsqueeze(1)
            score = score.masked_fill(mask == 0, -1e9)
        score = F.softmax(score, dim=-1)
        if dropout is not None:
            score = dropout(score)
        output += LayerNorm(score) + q
        fenmu = 0
    return output


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 多头注意力计算 (Full Self-Attention)
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


# 多头局部注意力计算 (Local-Attention)
class MultiHeadLocalAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # 创建q k列表用作local attention计算
        q_list, k_list = converse_qk(q, k)
        # calculate attention using function we will define next
        scores = local_attention(q_list, k_list, q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        del q_list, k_list
        return output


# 多头交叉注意力计算(Cross-Attention)
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = cross_attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


# FFN
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


# DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


# LocalAttentionLayer
class LocalAttentionLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.localattn = MultiHeadLocalAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.localattn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


# CrossAttentionLayer
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.localattn = MultiHeadLocalAttention(heads, d_model, dropout=dropout)
        self.crossattn = MultiHeadCrossAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, sa_outputs, pool_outputs, src_mask=None, trg_mask=None):
        x2 = self.norm_1(sa_outputs)
        x = sa_outputs + self.dropout_1(self.localattn(x2, x2, x2, src_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.crossattn(x2, pool_outputs, pool_outputs, trg_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


# 为了方便叠加 使用深复制
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


# Transformer 用于对比测试
# class Transformer(nn.Module):
#     def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
#         super().__init__()
#         self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
#         self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
#         self.out = nn.Linear(d_model, trg_vocab)
#
#     def forward(self, src, trg, src_mask, trg_mask):
#         e_outputs = self.encoder(src, src_mask)
#         d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
#         output = self.out(d_output)
#         return output


# ——————————————————————————————————————————————————————————————————————————————————————————————————————————
# TS-Left Net   size_window = Window, input_len = src.size(-1)
class BottomUp(nn.Module):
    def __init__(self, vocab_size, d_model, N_SelfA, N_LocalA, heads, dropout, size_window):
        super().__init__()
        self.N_SA = N_SelfA
        self.N_LocalA = N_LocalA
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.LocalAttnLayers = get_clones(LocalAttentionLayer(d_model, heads, dropout), N_LocalA)
        self.SelfAttnLayers = get_clones(EncoderLayer(d_model, heads, dropout), N_SelfA)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.size_window = size_window
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 1)

    def forward(self, src, mask, src_pad, input_len):
        x = self.embed(src)
        x = self.pe(x)
        # Local Self-Attention layers
        for i in range(self.N_LocalA):
            x = self.LocalAttnLayers[i](x, mask)
        # 保留第一次self-attention后的结果
        BUI_TRe = self.norm_1(x)
        # do the pooling of the output
        pooling = nn.AdaptiveAvgPool2d((input_len // self.size_window, self.d_model))
        x = self.norm_1(pooling(x))
        # 对pooling输出对纬度做一次映射 使变换纬度
        tem_x = self.linear(x).squeeze(-1)
        # 需要对输入重新进行一次mask操作 因为纬度不一样
        src_mask = (tem_x != src_pad).unsqueeze(-2)
        del tem_x
        # 进行Self-Attention操作
        for i in range(self.N_SA):
            x = self.SelfAttnLayers[i](x, src_mask)
        TL_Re = self.norm_2(x)
        return BUI_TRe, TL_Re


class TopDown(nn.Module):
    def __init__(self, vocab_size, d_model, N_Cross, heads, dropout):
        super().__init__()
        self.N_Cross = N_Cross
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.crossattnlayers = get_clones(CrossAttentionLayer(d_model, heads, dropout), N_Cross)
        self.norm = Norm(d_model)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, botttomup_infer, toplevel_repre):
        # 此处针对重新输入的数据的mask仍未完成
        # Local Self-Attention layers
        for i in range(self.N_Cross):
            x = self.crossattnlayers[i](botttomup_infer, toplevel_repre)
        # do the pooling of the output
        output = self.norm(x)
        return output


class TSnet(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N_full, N_local, N_cross, N_layer, heads, dropout, size_window,
                 src_pad):
        super().__init__()
        self.bottomup = BottomUp(src_vocab, d_model, N_full, N_local, heads, dropout, size_window)
        self.topdown = TopDown(src_vocab, d_model, N_cross, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N_layer, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
        self.src_pad = src_pad

    def forward(self, src, trg, src_mask, trg_mask):
        input_len = src.size(-1)
        BUI_TRe, TL_Re = self.bottomup(src, src_mask, self.src_pad, input_len)
        crossoutputs = self.topdown(BUI_TRe, TL_Re)
        d_output = self.decoder(trg, crossoutputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


# ————————————————————————————————————————————————训练代码——————————————————————————————————————————————————————
import time

Src_Data = 'data/cnn_test1.txt'
Trg_Data = 'data/cnn_test2.txt'
Src_Lang = 'en_core_web_sm'
Trg_Lang = 'en_core_web_sm'


# 创建模型 input:
def get_moduel(src_vocab, trg_vocab, src_pad):
    assert D_Model % Heads == 0
    assert Dropout < 1

    # model = Transformer(src_vocab, trg_vocab, D_Model, N_Layers, Heads, Dropout).to(torch.device('mps'))
    model = TSnet(src_vocab, trg_vocab, D_Model, N_FULL, N_local, N_cross, N_Layers, Heads, Dropout, Window, src_pad)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    module = model.to(Device)

    return module


def train_model(model, epochs, checkpoint, learnning_rate, train_data, train_len, printevery, src_pad, trg_pad):
    print("training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learnning_rate, betas=(0.9, 0.98), eps=1e-9)
    model.train()
    start = time.time()
    if checkpoint > 0:
        cptime = time.time()
    for epoch in range(epochs):
        total_loss = 0
        # 处理tqdm无法显示进度条的问题: 给一个total参数用来显示总长度
        for i, batch in tqdm(enumerate(train_data), total=train_len):

            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=trg_pad)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % printevery == 0:
                p = int(100 * (i + 1) / train_len)
                avg_loss = total_loss / printevery
                print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                      ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)),
                       "".join(' ' * (20 - (p // 5))), p, avg_loss))
                total_loss = 0

            if checkpoint > 0 and ((time.time() - cptime) // 60) // checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()

        # print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % \
        #       ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100,
        #        avg_loss, epoch + 1, avg_loss))
        print("%dm: epoch %d [%s%s] %d%%    loss = %.3f" % (
            (time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100,
            loss.item()))


def do_train():
    raw_src_data, raw_trg_data = read_data(Src_Data, Trg_Data)
    SRC, TRG = create_fields(Src_Lang, Trg_Lang)
    train_dataset, src_pad, trg_pad, train_len = create_dataset(raw_src_data, raw_trg_data, SRC, TRG)
    module = get_moduel(len(SRC.vocab), len(TRG.vocab), src_pad)
    train_model(module, Epochs, CheckPoint, LearningRate, train_dataset, train_len, PrintEvery, src_pad, trg_pad)


# test function of local-attention
def test_all():
    # src = torch.randint(low=3, high=1500, size=(1, 998)).to(Device)
    # trg = torch.randint(low=3, high=1500, size=(1, 54)).to(Device)
    # src_mask, trg_mask = create_masks(src, trg, 1, 1)
    # # 测试bottom-up-Module
    # #  def __init__(self, vocab_size, d_model, heads, n_full, n_local, size_window, input_len, dropout=0.1):
    # bottomupmodule = BottomUpInference(1000, D_Model, Heads, N_FULL, N_local, Window, src.size(-1), Dropout).to(Device)
    # output, bottomup_output, pooling_output = bottomupmodule(src, src_mask, 1)
    # calculate_A_ij(output, bottomup_output, pooling_output, D_Model)
    # q=torch.randn(2,1,736,512)
    # k=torch.randn(2,1,339,512)
    # v=torch.randn(2,1,339,512)
    # output = cross_attention(q,k,v,D_Model)
    # print(output)

    # sa_out= torch.randn(1,1296,512)
    # pool_out=torch.rand(1,403,512)
    # # src_mask, trg_mask = create_masks(pool_out, sa_out, 1, 1)
    # crossattentionlayer = CrossAttentionLayer(D_Model, Heads, Dropout)
    # out = crossattentionlayer(sa_out, pool_out, src_mask=None, trg_mask=None)
    # print(out.size())

    src = torch.randint(low=3, high=1500, size=(1, 998)).to(Device)
    trg = torch.randint(low=3, high=1500, size=(1, 54)).to(Device)
    src_mask, trg_mask = create_masks(src, trg, 1, 1)
    textsum = TSnet(1000, 600, D_Model, N_FULL, N_local, N_cross, N_Layers, Heads, Dropout, Window, src.size(-1), 1,
                    1).to(Device)
    out = textsum(src, trg, src_mask, trg_mask)
    print(out.size())


if __name__ == '__main__':
    do_train()
    # test_all()
