import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from opts import vocab_path


class Vocab:
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.word2idx = {}
        self.words = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for index, word in enumerate(f):
                w = word.strip('\n')
                self.word2idx[w] = index
                self.words.append(w)

    def __getitem__(self, item):
        return self.word2idx.get(item, self.word2idx.get(Vocab.UNK))

    def __len__(self):
        return len(self.words)


def build_vocab(vocab_path):
    return Vocab(vocab_path)


class LoadDataset:
    def __init__(self,
                 vocab_path,
                 batch_size=32,
                 split_sep='\n',
                 max_position_len=512,
                 pad_index=0,
                 shuffle=True
                 ):
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path).tokenize
        self.vocab = build_vocab(vocab_path)
        self.PAD_index = pad_index
        self.SEP_index = self.vocab['[SEP]']
        self.CLS_index = self.vocab['[CLS]']
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_position_len = max_position_len
        self.split_sep = split_sep

    def text_process(self, text):
        tmp = [self.CLS_index] + [self.vocab[w] for w in self.tokenizer(text)]
        tmp.append(self.SEP_index)
        tensor_text = torch.tensor(tmp, dtype=torch.long)
        tensor_text = tensor_text.unsqueeze(0)
        return tensor_text

    def data_process(self, file_path):
        raw_data = open(file_path, encoding='utf-8').readlines()
        data = []
        for raw in tqdm(raw_data, ncols=80):
            line = raw.rstrip("\n").split(self.split_sep)
            sentence, label = line[0], line[1]
            tmp = [self.CLS_index] + [self.vocab[w] for w in self.tokenizer(sentence)]
            if len(tmp) > self.max_position_len - 1:
                tmp = tmp[:self.max_position_len - 1]
            tmp.append(self.SEP_index)
            tensor_sentence = torch.tensor(tmp, dtype=torch.long)
            tensor_label = torch.tensor(int(label), dtype=torch.long)
            data.append((tensor_sentence, tensor_label))
        return data

    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (sentence, label) in data_batch:
            batch_sentence.append(sentence)
            batch_label.append(label)
        batch_sentence = seq_padding(batch_sentence)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label

    def dataloader(self, train_path, valid_path):
        train_data = self.data_process(train_path)
        test_data = self.data_process(valid_path)
        train_iter = DataLoader(train_data, batch_size=self.batch_size, shuffle=self.shuffle,
                                collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=self.batch_size, shuffle=self.shuffle,
                               collate_fn=self.generate_batch)
        return train_iter, test_iter


def seq_padding(X, padding=0):
    max_len = max([len(x) for x in X])
    out_tensor = []
    for x in X:
        if len(x) < max_len:
            x = torch.cat([x, torch.tensor([padding] * (max_len - len(x)))], dim=0)
        else:
            x = x[:max_len]
        out_tensor.append(x)
    out_tensor = torch.stack(out_tensor, dim=0)
    return out_tensor


load_data = LoadDataset(vocab_path=vocab_path,
                        split_sep='_!_')




