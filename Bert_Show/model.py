import os
from copy import deepcopy
from torch import nn
import torch
from opts import bert_pretrained_model_dir
from config import config


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(WordEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len]
        :return: [batch_size, seq_len, hidden_size]
        """
        out = self.embedding(x)
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_size, max_position_len=512):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_position_len, hidden_size)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len]
        :return: [batch_size, seq_len, hidden_size]
        """
        out = self.embedding(x)
        return out


class SegmentEmbedding(nn.Module):
    def __init__(self, segment_size, hidden_size):
        super(SegmentEmbedding, self).__init__()
        self.embedding = nn.Embedding(segment_size, hidden_size)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len]
        :return: [batch_size, seq_len, hidden_size]
        """
        out = self.embedding(x)
        return out


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_len, segment_size, hidden_dropout_prob=0.1):
        super(BertEmbedding, self).__init__()
        self.word_embedding = WordEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size
        )

        self.position_embedding = PositionalEmbedding(
            max_position_len=max_position_len,
            hidden_size=hidden_size
        )

        self.segment_embedding = SegmentEmbedding(
            segment_size=segment_size,
            hidden_size=hidden_size
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(max_position_len).expand((1, -1)))

    def forward(self, word_inputs, position_inputs=None, segment_inputs=None):
        """
        :param word_inputs: [batch_size, seq_len]
        :param position_inputs: [batch_size, seq_len]
        :param segment_inputs: [batch_size, seq_len]
        :return: [batch_size, seq_len, hidden_size]
        """
        src_len = word_inputs.size(1)
        word_embedding = self.word_embedding(word_inputs)

        if position_inputs is None:
            position_inputs = self.position_ids[:, :src_len]
        position_embedding = self.position_embedding(position_inputs)

        if segment_inputs is None:
            segment_inputs = torch.zeros_like(word_inputs, device=self.position_ids.device)
        segment_embedding = self.segment_embedding(segment_inputs)

        embeddings = word_embedding + position_embedding + segment_embedding
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, hidden_size, dropout_prob=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        self.qw = nn.Linear(hidden_size, hidden_size)
        self.kw = nn.Linear(hidden_size, hidden_size)
        self.vw = nn.Linear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, attn_mask=None):
        """
        :param x: [batch_size, seq_len, hidden_size]
        :return: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.size()
        q, k, v = self.qw(x), self.kw(x), self.vw(x)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attention = q @ k.transpose(-1, -2) * self.scaling
        attn_mask = attn_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention = attention + attn_mask
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)

        output = attention @ v
        output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        output = self.linear(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_prob=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.act_fun = nn.GELU()
        self.linear_2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        output = self.linear_1(x)
        output = self.act_fun(output)
        output = self.linear_2(output)
        output = self.dropout(output)
        return output


class BertLayer(nn.Module):
    def __init__(self, num_heads, hidden_size, attention_prob=0.1, hidden_dropout_prob=0.1, attn_mask=None):
        super(BertLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(
            num_heads=num_heads,
            hidden_size=hidden_size,
            dropout_prob=attention_prob,
            attn_mask=attn_mask
        )
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            dropout_prob=hidden_dropout_prob
        )
        self.norm_2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, attn_mask=None):
        """
        :param x:
        :return:
        """
        output = self.norm_1(x + self.attention(x, attn_mask))
        output = self.norm_2(output + self.feed_forward(output))
        return output


class BertOutput(nn.Module):
    def __init__(self, hidden_size):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        output = self.dense(x)
        return output


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.config = config
        self.embedding = BertEmbedding(vocab_size=config.vocab_size,
                                       hidden_size=config.hidden_size,
                                       max_position_len=config.max_position_len,
                                       segment_size=config.segment_size,
                                       hidden_dropout_prob=config.hidden_dropout_prob)
        self.layers = nn.ModuleList([
            BertLayer(num_heads=config.num_heads,
                      hidden_size=config.hidden_size,
                      attention_prob=config.attention_prob,
                      hidden_dropout_prob=config.hidden_dropout_prob,
                      attn_mask=None)
            for _ in range(config.num_layers)
        ])
        self.head = BertOutput(hidden_size=config.hidden_size)

    def forward(self, x, attn_mask=None):
        """
        :param x:
        :return:
        """
        x = self.embedding(x)
        all_layers_outputs = []
        for layer in self.layers:
            x = layer(x, attn_mask)
            all_layers_outputs.append(x)
        output = self.head(x)
        return output, all_layers_outputs

    @classmethod
    def from_pretrained(cls, config, pretrained_model_dir=None):
        model = cls(config)
        pretrained_model_file = os.path.join(pretrained_model_dir, 'pytorch_model.bin')
        pretrained_model = torch.load(pretrained_model_file)
        state_dict = deepcopy(model.state_dict())
        pretrained_model_list = list(pretrained_model.keys())[:-8]
        model_list = list(state_dict.keys())[1:]
        for i in range(len(pretrained_model_list)):
            state_dict[model_list[i]] = pretrained_model[pretrained_model_list[i]]
        model.load_state_dict(state_dict)
        return model


def bert_model(config, pretrained_model_dir=None):
    model = Bert(config)
    if pretrained_model_dir:
        model = Bert.from_pretrained(config, pretrained_model_dir)
    return model


def make_model():
    return bert_model(config, bert_pretrained_model_dir)


# downstream tasks
class BertForTextClassifier(nn.Module):
    def __init__(self):
        super(BertForTextClassifier, self).__init__()
        self.Bert = make_model()
        self.dropout = nn.Dropout(self.Bert.config.hidden_dropout_prob)
        self.num_labels = self.Bert.config.num_classes
        self.classifier = nn.Linear(self.Bert.config.hidden_size, self.num_labels)

    def forward(self, x, attn_mask=None):
        output, _ = self.Bert(x, attn_mask)
        output = self.dropout(output)
        output = self.classifier(output[:, 0])
        return output
