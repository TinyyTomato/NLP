from torch import nn
import json
from opts import json_file


class Config(nn.Module):
    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 num_layers=12,
                 num_heads=12,
                 max_position_len=512,
                 segment_size=2,
                 attention_prob=0.1,
                 hidden_dropout_prob=0.1,
                 num_classes=15
                 ):
        super(Config, self).__init__()
        self.vocab_size = vocab_size
        self.max_position_len = max_position_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.segment_size = segment_size
        self.num_layers = num_layers
        self.attention_prob = attention_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_classes = num_classes

    @classmethod
    def from_dict(cls, json_dict):
        config = Config()
        for key, value in json_dict.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, 'r') as f:
            json_dict = f.read()
        return cls.from_dict(json.loads(json_dict))


config = Config.from_json(json_file)