import pandas as pd
import torch
from dataloader import load_data

checkpoint_path = './data/checkpoint_best_model.pth'
model = torch.load(checkpoint_path, map_location='cpu')
model.eval()
name_list = ["民生 故事", "文化 文学", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际",
             "证券 股票", "农业 三农", "电竞 游戏"]


def text_classification(text):
    """
    This function takes a text input and returns the predicted class.
    """
    data = load_data.text_process(text)
    pad_mask = (data == 0)
    label = model(data, pad_mask)
    label = label.argmax(dim=1).item()
    return name_list[label]

while (1):
    text = input()
    result = text_classification(text)
    print(result)
