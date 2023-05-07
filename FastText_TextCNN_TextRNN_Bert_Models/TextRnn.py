# TextRNN（循环神经网络）进行文本特征抽取，由于文本本身是一种序列，而LSTM天然适合建模序列数据。
# TextRNN将句子中每个词的词向量依次输入到双向双层LSTM，分别将两个方向最后一个有效位置的隐藏层拼接成一个向量作为文本的表示。
# TextCNN利用CNN（卷积神经网络）进行文本特征抽取，不同大小的卷积核分别抽取n-gram特征，
# 卷积计算出的特征图经过MaxPooling保留最大的特征值，然后将拼接成一个向量作为文本的表示。
# 这里我们基于TextCNN原始论文的设定，分别采用了100个大小为2,3,4的卷积核，最后得到的文本向量大小为100*3=300维。

import numpy as np
import pandas as pd
# 利用pytorch手写TextCNN
import torch
# 显示运行出...的时间, 便于调整模型的时间复杂度
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

import random
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

# 配置使用GPU运行代码
gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")
logging.info("Use cuda: %s, gpu id: %d.", use_cuda, gpu)

# 将数据分为10份
fold_num = 10
data_file = 'train_set.csv'

# 将数据转化为份数
def all_data2fold(fold_num, num=10000):
    fold_data = []
    f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    # 获取text内容和label标签
    texts = f['text'].tolist()[:num]
    labels = f['label'].tolist()[:num]
    # 总数
    total = len(labels)
    index = list(range(total))
    # 数据清洗(混打)
    np.random.shuffle(index)
    # 存放text和标签
    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])
    # 建立一个label与id一一对应的字典
    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)
    # 把这一份fold_num中的数据读进来
    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        # print(label, len(data))
        # 设置batch_size长度, 向下取整
        batch_size = int(len(data) / fold_num)
        # 整数个batch_size以外的batch_size大小
        other = len(data) - batch_size * fold_num
        for i in range(fold_num):
            # 计算当前的cur_batch_size大小
            cur_batch_size = batch_size + 1 if i < other else batch_size
            # print(cur_batch_size)
            # 将一维数据整理成二维平面, 成为真正的batch_data的内容
            batch_data = [data[i * batch_size + b] for b in range(cur_batch_size)]
            # 在对应的index数据后追加有关于batch_data的数据
            all_index[i].extend(batch_data)

    # 与上同理, batch_size的设置总数/份数并向下取整
    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0
    # 对于每一份数中的每一份数据来说
    for fold in range(fold_num):
        # 记录数据长度
        num = len(all_index[fold])
        # 记录texts和labels数据
        texts = [all_texts[i] for i in all_index[fold]]
        labels = [all_labels[i] for i in all_index[fold]]

        if num > batch_size:
            # num大于batch_size代表进去的batch_data超过了应有的长度, 需要使用other来对剩余的bacth_data数据进行处理
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:
            # num大于batch_size代表进去的batch_data小于应有的长度, 表明数据来到了末尾, 处理剩下的情况, 则这个batch_data直接把剩余的batch_data作为一个batch
            end = start + batch_size - num
            fold_texts = texts + other_texts[start: end]
            fold_labels = labels + other_labels[start: end]
            start = end
        else:
            # 相等的话直接赋值
            fold_texts = texts
            fold_labels = labels

        # 判断batch_size的大小是否等于fold_labels的长度, 保证程序运行的正确性
        assert batch_size == len(fold_labels)

        # 随机清洗打乱原有的batch
        index = list(range(batch_size))
        np.random.shuffle(index)

        # 记录清洗打乱过后的texts和labels
        shuffle_fold_texts = []
        shuffle_fold_labels = []
        for i in index:
            shuffle_fold_texts.append(fold_texts[i])
            shuffle_fold_labels.append(fold_labels[i])

        # 用data字典存放打乱之后的数据
        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
        fold_data.append(data)
    # 输出每一个fold的长度
    logging.info("Fold lens %s", str([len(data['label']) for data in fold_data]))
    return fold_data

# 把数据拆成十份
fold_data = all_data2fold(10)

# 建立训练集、验证集与测试集
fold_id = 9
# 验证集
dev_data = fold_data[fold_id]
# 训练集
train_texts = []
train_labels = []
for i in range(0, fold_id):
    # 获取训练集数据, 并记录训练集的内容与标签
    data = fold_data[i]
    train_texts.extend(data['text'])
    train_labels.extend(data['label'])
# 用字典将text和label对应存起来
train_data = {'label': train_labels, 'text': train_texts}

# 测试集
test_data_file = 'test_a.csv'
f = pd.read_csv(test_data_file, sep='\t', encoding='UTF-8')
# 将测试集中的内容从csv的df转化为list
texts = f['text'].tolist()
# 测试集的label位于text的第零列, 将测试集的label和text存入test_data中
test_data = {'label': [0] * len(texts), 'text': texts}

# 建立词库
# counter()函数返回的是一个类似于字典的counter计数器
from collections import Counter
from transformers import BasicTokenizer
# 这个方法用来转换为unicode编码, 标点符号分割、小写转换、中文字符分割、去除重音符号等操作, 最后返回的是关于词的数组
basic_tokenizer = BasicTokenizer()

# 词库
class Vocab():
    def __init__(self, train_data):
        # 一个词最小出现次数是5
        self.min_count = 5
        # pad是补0字符, unk是其他低频率的单词
        self.pad = 0
        self.unk = 1
        self._id2word = ['[PAD]', '[UNK]']
        self._id2extword = ['[PAD]', '[UNK]']
        self._id2label = []
        self.target_names = []
        self.build_vocab(train_data)

        # zip函数的功能是将可迭代的对象作为参数, 将对象中对应的元素打包成一个个元组, 然后返回由这些元组组成的zip对象
        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        self._label2id = reverse(self._id2label)

        # 打印出词库信息: 一共多少词, text一共多少个类别
        logging.info("Build vocab: words %d, labels %d." % (self.word_size, self.label_size))

    # 建立词库
    def build_vocab(self, data):
        self.word_counter = Counter()
        # 通过将text文本中的单词分词, 计算每个单词出现的个数(词频)
        for text in data['text']:
            words = text.split()
            for word in words:
                self.word_counter[word] += 1

        # 如果一个单词比出现的最小频率高, 那就意味着这个单词不是低频率单词, 是一个可以加入计算的单词
        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        # 标签的名字(一共14个类别, 恰好印证了前面的计算结果正确性)
        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育', 7: '财经',
                      8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}
        self.label_counter = Counter(data['label'])

        # 计数label的count与target_names
        # _id2label = _id to label
        for label in range(len(self.label_counter)):
            count = self.label_counter[label]
            self._id2label.append(label)
            self.target_names.append(label2name[label])

    # 定义一个加载预训练的嵌入层
    def load_pretrained_embs(self, embfile):
        with open(embfile, encoding='utf-8') as f:
            lines = f.readlines()
            items = lines[0].split()
            # 第一行分别是单词数量、词向量维度
            word_count, embedding_dim = int(items[0]), int(items[1])

        # np.zero函数返回来一个给定形状和类型的用0填充的数组, 目的是将每句句子进行消齐操作, 传入embedding层的句子一定是等维度的
        # self._id2extword = ['[PAD]', '[UNK]']
        # _id2extword = _id to extword
        index = len(self._id2extword)
        # 用零填充长度不足的句子
        embeddings = np.zeros((word_count + index, embedding_dim))
        for line in lines[1:]:
            values = line.split()
            self._id2extword.append(values[0])
            vector = np.array(values[1:], dtype='float64')
            embeddings[self.unk] += vector
            embeddings[index] = vector
            index += 1

        # unk是其他低频率的单词
        embeddings[self.unk] = embeddings[self.unk] / word_count
        # embedding层标准化
        embeddings = embeddings / np.std(embeddings)

        # lambda是匿名函数, 自己定义一个函数执行规则(函数执行体)
        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        assert len(set(self._id2extword)) == len(self._id2extword)
        return embeddings

    # 根据 XXX 得到 id
    # 将单词转换为其对应的id
    def word2id(self, xs):
        # isinstance函数用于判断对象类型是否符合预期
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)

    # 处理_extword to id
    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)

    # 处理_label to id
    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    # @property的作用:
    # 1、修饰方法, 是方法可以像属性一样访问
    # 2、与所定义的属性配合使用, 这样可以防止属性被修改。
    @property
    def word_size(self):
        return len(self._id2word)

    # self._id2extword = ['[PAD]', '[UNK]']
    @property
    def extword_size(self):
        return len(self._id2extword)

    @property
    def label_size(self):
        return len(self._id2label)

# 基于train_data建立自己的词库
vocab = Vocab(train_data)

'''
Vocab 的作用是：
1. 创建 词 和 index 对应的字典，这里包括 2 份字典，分别是：_id2word 和 _id2extword
其中 _id2word 是从新闻得到的， 把词频小于 5 的词替换为了 UNK。对应到模型输入的 batch_inputs1。
_id2extword 是从 word2vec.txt 中得到的，有 5976 个词。对应到模型输入的 batch_inputs2。
后面会有两个 embedding 层，其中 _id2word 对应的 embedding 是可学习的，
_id2extword 对应的 embedding 是从文件中加载的，是固定的
2.创建 label 和 index 对应的字典
'''

# 建立模型
import torch.nn as nn
import torch.nn.functional as F

# 自注意力机制
'''
Attention的输入是sent_hiddens和sent_masks。
在Attention里，sent_hiddens首先经过线性变化得到key，维度不变，依然是(batch_size , doc_len, 512)。
然后key和query相乘，得到outputs。query的维度是512，因此output的维度是(batch_size , doc_len)，
这个就是我们需要的attention，表示分配到每个句子的权重。
下一步需要对这个attetion做softmax，并使用sent_masks，把没有单词的句子的权重置为-1e32，得到masked_attn_scores。
最后把masked_attn_scores和key相乘，得到batch_outputs，形状是(batch_size, 512)。
'''

class Attention(nn.Module):
    def __init__(self, hidden_size):
        # super()函数的目的是: 解决多重继承时父类的查找问题
        super(Attention, self).__init__()
        # 权重
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # normal_函数返回一个指定区间内的随机生成正态分布的值
        self.weight.data.normal_(mean=0.0, std=0.05)

        # 偏置
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        # b的大小根据hidden_size的大小确定, 不足的用32位浮点数的0补齐
        b = np.zeros(hidden_size, dtype=np.float32)
        # 复制一份保留
        self.bias.data.copy_(torch.from_numpy(b))

        self.query = nn.Parameter(torch.Tensor(hidden_size))
        self.query.data.normal_(mean=0.0, std=0.05)

    def forward(self, batch_hidden, batch_masks):
        # 前向传播函数
        # batch_hidden: b x len x hidden_size = 2 * hidden_size of lstm
        # batch_masks:  b x len
        # masks代表看不见的单词[masked language model]

        # 线性输入层linear
        # torch.matmul计算两个矩阵相乘, 即点乘得到的结果(会进行广播)
        key = torch.matmul(batch_hidden, self.weight) + self.bias
        # batch_hidden = b x len x hidden

        # 计算注意力模块
        # 大小是: b x len
        outputs = torch.matmul(key, self.query)
        # masked_fill: 其作用是在为 1 的地方替换为 value: float(-1e32)
        masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e32))
        # attn_scores(attention 分数)的大小是: b x len
        attn_scores = F.softmax(masked_outputs, dim=1)
        # 对于全零向量, -1e32的结果为 1/len. -inf为nan(实际是无穷大), 需要额外补0
        masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)

        # 对分数进行加权求和
        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)  # b x hidden
        return batch_outputs, attn_scores

# build word encoder
word2vec_path = './bert/bert-mini/word2vec.txt'
# 在训练时, 丢失的神经元比例是15%
dropout = 0.15
# 隐藏层神经元个数+层数设置
word_hidden_size = 128
word_num_layers = 2

class WordLSTMEncoder(nn.Module):
    def __init__(self, vocab):
        super(WordLSTMEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.word_dims = 100

        self.word_embed = nn.Embedding(vocab.word_size, self.word_dims, padding_idx=0)

        extword_embed = vocab.load_pretrained_embs(word2vec_path)
        extword_size, word_dims = extword_embed.shape
        logging.info("Load extword embed: words %d, dims %d." % (extword_size, word_dims))

        self.extword_embed = nn.Embedding(extword_size, word_dims, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        self.extword_embed.weight.requires_grad = False

        input_size = self.word_dims

        self.word_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=word_hidden_size,
            num_layers=word_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, word_ids, extword_ids, batch_masks):
        # word_ids: sen_num x sent_len
        # extword_ids: sen_num x sent_len
        # batch_masks   sen_num x sent_len

        # sen_num x sent_len x 100
        word_embed = self.word_embed(word_ids)
        extword_embed = self.extword_embed(extword_ids)
        batch_embed = word_embed + extword_embed

        if self.training:
            batch_embed = self.dropout(batch_embed)
        # sen_num x sent_len x  hidden * 2(基于两个LSTM的隐藏层)
        hiddens, _ = self.word_lstm(batch_embed)
        hiddens = hiddens * batch_masks.unsqueeze(2)

        if self.training:
            batch_embed = self.dropout(batch_embed)
        # sen_num x sent_len x  hidden * 2(基于两个LSTM的隐藏层)
        hiddens, _ = self.word_lstm(batch_embed)
        hiddens = hiddens * batch_masks.unsqueeze(2)
        if self.training:
            hiddens = self.dropout(hiddens)
        return hiddens

# 建立传输编码层
sent_hidden_size = 256
sent_num_layers = 2
'''
定义 SentEncoder
SentEncoder包含了 2 层的双向 LSTM，输入数据sent_reps的形状是(batch_size , doc_len, 300)，
LSTM 的 hidden_size 为 256，由于是双向的，经过 LSTM 后的数据维度是(batch_size , doc_len, 512)，
然后和 mask 按位置相乘，把没有单词的句子的位置改为 0，
最后输出的数据sent_hiddens，维度依然是(batch_size , doc_len, 512)。
'''

class SentEncoder(nn.Module):
    def __init__(self, sent_rep_size):
        super(SentEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 这里直接调用LSTM的库, 使用LSTM建立传输层
        self.sent_lstm = nn.LSTM(
            input_size=sent_rep_size,
            hidden_size=sent_hidden_size,
            num_layers=sent_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, sent_reps, sent_masks):
        # 前向传播中的各变量名维度大小如下:
        # sent_reps:  b x doc_len x sent_rep_size
        # sent_masks: b x doc_len

        # b x doc_len x hidden * 2(与之前的两个LSTM相对应)
        sent_hiddens, _ = self.sent_lstm(sent_reps)
        # 对应相乘, 用到广播, 是为了只保留有句子的位置的数值
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)
        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)
        return sent_hiddens



# 建立模型
class Model(nn.Module):
    def __init__(self, vocab):
        super(Model, self).__init__()
        self.sent_rep_size = word_hidden_size * 2
        self.doc_rep_size = sent_hidden_size * 2
        self.all_parameters = {}

        parameters = []
        # sent-encoder部分使用lstm
        self.word_encoder = WordLSTMEncoder(vocab)
        self.word_attention = Attention(self.sent_rep_size)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.word_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.word_attention.parameters())))

        self.sent_encoder = SentEncoder(self.sent_rep_size)
        self.sent_attention = Attention(self.doc_rep_size)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))
        self.out = nn.Linear(self.doc_rep_size, vocab.label_size, bias=True)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))

        if use_cuda:
            self.to(device)
        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters
        logging.info('Build model with lstm word encoder, lstm sent encoder.')
        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        logging.info('Model param num: %.2f M.' % (para_num / 1e6))

    def forward(self, batch_inputs):
        # 前向传播, 各参数变量维度大小如下:
        # batch_inputs(batch_inputs1, batch_inputs2): b x doc_len x sent_len
        # batch_masks : b x doc_len x sent_len
        # batch_masks 就是取反, 把没有单词的句子置为 0

        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[0], batch_inputs1.shape[1], batch_inputs1.shape[2]
        # sen_num x sent_len
        batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len, max_sent_len)
        # sen_num x sent_len
        batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len, max_sent_len)
        # sen_num x sent_len
        batch_masks = batch_masks.view(batch_size * max_doc_len, max_sent_len)
        # sen_num x sent_len x sent_rep_size
        batch_hiddens = self.word_encoder(batch_inputs1, batch_inputs2, batch_masks)
        # sen_num x sent_rep_size
        sent_reps, atten_scores = self.word_attention(batch_hiddens, batch_masks)
        # b x doc_len x sent_rep_size
        sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)
        # b x doc_len x max_sent_len
        batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)
        # b x doc_len
        sent_masks = batch_masks.bool().any(2).float()
        # b x doc_len x doc_rep_size
        sent_hiddens = self.sent_encoder(sent_reps, sent_masks)
        # b x doc_rep_size
        doc_reps, atten_scores = self.sent_attention(sent_hiddens, sent_masks)
        # b x num_labels
        batch_outputs = self.out(doc_reps)
        return batch_outputs
model = Model(vocab)

# 建立优化器
# 各参数的设置
learning_rate = 2e-4
# 学习率衰减因子
decay = .75
# 学习率衰减步长
decay_step = 1000

class Optimizer:
    def __init__(self, model_parameters):
        self.all_params = []
        self.optims = []
        self.schedulers = []

        for name, parameters in model_parameters.items():
            if name.startswith("basic"):
                # 如果开始带着basic, 那就开始优化
                optim = torch.optim.Adam(parameters, lr=learning_rate)
                self.optims.append(optim)
                # //的意思是向下取整
                l = lambda step: decay ** (step // decay_step)
                # torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)
                # 设置学习率为初始学习率乘以给定lr_lambda函数的值
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=l)
                self.schedulers.append(scheduler)
                self.all_params.extend(parameters)
            else:
                Exception("no nameed parameters.")
        self.num = len(self.optims)

    def step(self):
        # 步长
        for optim, scheduler in zip(self.optims, self.schedulers):
            optim.step()
            scheduler.step()
            optim.zero_grad()

    def zero_grad(self):
        # 零梯度下降
        for optim in self.optims:
            optim.zero_grad()

    def get_lr(self):
        # learning_rate重新计算
        lrs = tuple(map(lambda x: x.get_lr()[-1], self.schedulers))
        lr = ' %.5f' * self.num
        res = lr % lrs
        return res

'''
定义 sentence_split，把文章划分为句子
输入的text表示一篇新闻，最后返回的 segments 是一个list，其中每个元素是 tuple：(句子长度，句子本身)。

作用是：根据一篇文章，把这篇文章分割成多个句子
text 是一个新闻的文章
vocab 是词典
max_sent_len 表示每句话的长度
max_segment 表示最多有几句话
最后返回的 segments 是一个list，其中每个元素是 tuple：(句子长度，句子本身)
'''

# 建立数据集
def sentence_split(text, vocab, max_sent_len=256, max_segment=16):
    # 句子分割
    words = text.strip().split()
    document_len = len(words)
    index = list(range(0, document_len, max_sent_len))
    index.append(document_len)

    segments = []
    for i in range(len(index) - 1):
        segment = words[index[i]: index[i + 1]]
        # 确保每一部分的长度都是大于0的
        assert len(segment) > 0
        # 把出现太少的词替换为 UNK
        segment = [word if word in vocab._id2word else '<UNK>' for word in segment]
        segments.append([len(segment), segment])

    assert len(segments) > 0
    if len(segments) > max_segment:
        # 如果分得的segment比设置的最大segment还大, 那就将大的segment分割成两个小的segment
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[-segment_:]
    else:
        return segments

# 遍历每一篇新闻，对每篇新闻都调用sentence_split来分割句子。
# 最后返回的数据是一个 list，每个元素是一个 tuple: (label, 句子数量，doc)。
# 其中 doc 又是一个 list，每个 元素是一个 tuple: (句子长度，word_ids, extword_ids)。

def get_examples(data, vocab, max_sent_len=256, max_segment=8):
    # 获取数据集dataset中的例子
    label2id = vocab.label2id
    examples = []

    for text, label in zip(data['text'], data['label']):
        # 记录label
        id = label2id(label)
        # 记录sents_words
        sents_words = sentence_split(text, vocab, max_sent_len, max_segment)
        doc = []
        for sent_len, sent_words in sents_words:
            word_ids = vocab.word2id(sent_words)
            extword_ids = vocab.extword2id(sent_words)
            doc.append([sent_len, word_ids, extword_ids])
        examples.append([id, len(doc), doc])
    # 打印信息: 总共有多少个doc
    logging.info('Total %d docs.' % len(examples))
    return examples

# 把数据分割为多个 batch，组成一个 list 并返回
def batch_slice(data, batch_size):
    # 设置batch_num与记录每轮的docs
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]
        # yield: 带yield的函数是一个迭代器, 在函数内部碰到yield的时候, 函数会返回某个值, 并停留在这个位置; 当下次执行函数后, 会在上次停留的位置继续运行。
        yield docs

def data_iter(data, batch_size, shuffle=True, noise=1.0):
    # 数据迭代(器): 在迭代训练时，调用data_iter函数, 生成每一批的batch_data。
    # 而data_iter函数里面会调用batch_slice函数。
    # 随机排列数据, 然后按原长度排序, 分批一批批的, 保证每批句子的长度是一致的

    batched_data = []
    if shuffle:
        np.random.shuffle(data)
        lengths = [example[1] for example in data]
        # 噪声数据
        noisy_lengths = [- (l + np.random.uniform(- noise, noise)) for l in lengths]
        # argsort是numpy中的一个函数, 用来返回一个数组排好序后各元素对应的原来的位置序号。
        # 经过排序的结果
        sorted_indices = np.argsort(noisy_lengths).tolist()
        sorted_data = [data[i] for i in sorted_indices]
    else:
        sorted_data = data

    batched_data.extend(list(batch_slice(sorted_data, batch_size)))
    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch



# 计算模型得分的函数(库)
from sklearn.metrics import f1_score, precision_score, recall_score
def get_score(y_ture, y_pred):
    y_ture = np.array(y_ture)
    y_pred = np.array(y_pred)
    f1 = f1_score(y_ture, y_pred, average='macro') * 100
    p = precision_score(y_ture, y_pred, average='macro') * 100
    r = recall_score(y_ture, y_pred, average='macro') * 100
    return str((reformat(p, 2), reformat(r, 2), reformat(f1, 2))), reformat(f1, 2)

def reformat(num, n):
    return float(format(num, '0.' + str(n) + 'f'))

# 建立训练模型
# 输出模型训练时间
import time
from sklearn.metrics import classification_report
# 各参数的定义
clip = 5.0
epochs = 15
early_stops = 3
log_interval = 200
test_batch_size = 16
train_batch_size = 16
# 模型保存地址与保存训练结果
save_model = './rnn.bin'
save_test = './rnn.csv'

class Trainer():
    def __init__(self, model, vocab):
        self.model = model
        self.report = True
        self.train_data = get_examples(train_data, vocab)
        self.batch_num = int(np.ceil(len(self.train_data) / float(train_batch_size)))
        self.dev_data = get_examples(dev_data, vocab)
        self.test_data = get_examples(test_data, vocab)
        # 计算交叉熵损失
        self.criterion = nn.CrossEntropyLoss()
        # label name
        self.target_names = vocab.target_names
        self.optimizer = Optimizer(model.all_parameters)
        self.step = 0
        self.early_stop = -1
        self.best_train_f1, self.best_dev_f1 = 0, 0
        self.last_epoch = epochs

    def train(self):
        logging.info('Start training...')
        for epoch in range(1, epochs + 1):
            train_f1 = self._train(epoch)
            dev_f1 = self._eval(epoch)
            if self.best_dev_f1 <= dev_f1:
                logging.info("Exceed history dev = %.2f, current dev = %.2f" % (self.best_dev_f1, dev_f1))
                torch.save(self.model.state_dict(), save_model)
                self.best_train_f1 = train_f1
                self.best_dev_f1 = dev_f1
                self.early_stop = 0
            else:
                self.early_stop += 1
                if self.early_stop == early_stops:
                    # 输出模型过拟合的情况
                    logging.info("Eearly stop in epoch %d, best train: %.2f, dev: %.2f" %
                                 (epoch - early_stops, self.best_train_f1, self.best_dev_f1))
                    self.last_epoch = epoch
                    break

    def test(self):
        self.model.load_state_dict(torch.load(save_model))
        self._eval(self.last_epoch + 1, test=True)

    def _train(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()

        start_time = time.time()
        epoch_start_time = time.time()
        overall_losses = 0
        losses = 0
        batch_idx = 1
        y_pred = []
        y_true = []

        for batch_data in data_iter(self.train_data, train_batch_size, shuffle=True):
            torch.cuda.empty_cache()
            batch_inputs, batch_labels = self.batch2tensor(batch_data)
            batch_outputs = self.model(batch_inputs)
            loss = self.criterion(batch_outputs, batch_labels)
            loss.backward()
            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value
            # 把预测值转换为一维, 方便下面做 classification_report, 计算 f1
            y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.optimizer.all_params, max_norm=clip)

            for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
                optimizer.step()
                scheduler.step()
            self.optimizer.zero_grad()
            self.step += 1

            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time
                lrs = self.optimizer.get_lr()
                logging.info('| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr{} | loss {:.4f} | s/batch {:.2f}'
                    .format(epoch, self.step, batch_idx, self.batch_num, lrs, losses / log_interval, elapsed / log_interval))
                losses = 0
                start_time = time.time()
            batch_idx += 1
        overall_losses /= self.batch_num
        during_time = time.time() - epoch_start_time
        # 重格式化: reformat
        overall_losses = reformat(overall_losses, 4)
        score, f1 = get_score(y_true, y_pred)
        logging.info('| epoch {:3d} | score {} | f1 {} | loss {:.4f} | time {:.2f}'.format
                     (epoch, score, f1, overall_losses, during_time))
        # 如果预测和真实的标签都包含相同的类别数目, 才能调用 classification_report
        if set(y_true) == set(y_pred) and self.report:
            report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
            logging.info('\n' + report)
        return f1

    def _eval(self, epoch, test=False):
        # 作用同eval函数原理相同
        self.model.eval()
        start_time = time.time()
        data = self.test_data if test else self.dev_data
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_data in data_iter(data, test_batch_size, shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = self.batch2tensor(batch_data)
                batch_outputs = self.model(batch_inputs)
                y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())
            score, f1 = get_score(y_true, y_pred)
            during_time = time.time() - start_time
            if test:
                df = pd.DataFrame({'label': y_pred})
                df.to_csv(save_test, index=False, sep=',')
            else:
                logging.info('| epoch {:3d} | dev | score {} | f1 {} | time {:.2f}'
                             .format(epoch, score, f1, during_time))
                if set(y_true) == set(y_pred) and self.report:
                    report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
                    logging.info('\n' + report)
        return f1

    # batch2tensor函数最后返回的数据是：(batch_inputs1, batch_inputs2, batch_masks), batch_labels。
    # 形状都是(batch_size, doc_len, sent_len)。doc_len表示每篇新闻有几乎话，sent_len表示每句话有多少个单词。
    # batch_masks在有单词的位置，值为1，其他地方为0，用于后面计算Attention，把那些没有单词的位置的attention改为0。
    # batch_inputs1, batch_inputs2, batch_masks，形状是(batch_size, doc_len, sent_len)，
    # 转换为(batch_size * doc_len, sent_len)。

    def batch2tensor(self, batch_data):
        # 每个batch转换成词向量的格式如下图:
        # [[label, doc_len, [[sent_len, [sent_id0, ...], [sent_id1, ...]], ...]]
        batch_size = len(batch_data)
        doc_labels = []
        doc_lens = []
        doc_max_sent_len = []
        # 利用for循环将batch_data读入doc_data中
        for doc_data in batch_data:
            doc_labels.append(doc_data[0])
            doc_lens.append(doc_data[1])
            sent_lens = [sent_data[0] for sent_data in doc_data[2]]
            max_sent_len = max(sent_lens)
            doc_max_sent_len.append(max_sent_len)

        # 最大的单词长度, 最长的句子长度
        max_doc_len = max(doc_lens)
        max_sent_len = max(doc_max_sent_len)

        # 每个batch的情况(inputs + masks + labels)
        batch_inputs1 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_inputs2 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_masks = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.float32)
        batch_labels = torch.LongTensor(doc_labels)

        for b in range(batch_size):
            for sent_idx in range(doc_lens[b]):
                sent_data = batch_data[b][2][sent_idx]
                for word_idx in range(sent_data[0]):
                    batch_inputs1[b, sent_idx, word_idx] = sent_data[1][word_idx]
                    batch_inputs2[b, sent_idx, word_idx] = sent_data[2][word_idx]
                    batch_masks[b, sent_idx, word_idx] = 1
        if use_cuda:
            batch_inputs1 = batch_inputs1.to(device)
            batch_inputs2 = batch_inputs2.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)
        return (batch_inputs1, batch_inputs2, batch_masks), batch_labels

# train
trainer = Trainer(model, vocab)
trainer.train()

# test
trainer.test()
