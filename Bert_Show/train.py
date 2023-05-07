import os
from tqdm import tqdm
from model import BertForTextClassifier
from dataloader import load_data
from opts import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, test_data = load_data.dataloader(train_data_path, test_data_path)

filepath = os.path.join(checkpoint_path, 'checkpoint_best_model.pth')
if os.path.isfile(filepath):
    print("model loaded!")
    model = torch.load(filepath)
else:
    model = BertForTextClassifier()
model = model.to(device)
print(model.parameters)


loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_acc = 0

for epoch in range(num_epochs):
    accuracy_num = 0
    all_num = 0
    model.train()
    with tqdm(train_data, unit="batch") as tepoch:
        for sentences, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1} train")
            sentences = sentences.to(device)
            labels = labels.to(device)
            pad_mask = (sentences == 0)
            outputs = model(sentences, pad_mask)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())
    model.eval()
    with tqdm(test_data, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch + 1} test")
        for sentences, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1} test")
            sentences = sentences.to(device)
            labels = labels.to(device)
            pad_mask = (sentences == 0)
            outputs = model(sentences, pad_mask)
            accuracy_num += (outputs.argmax(1) == labels).float().sum().item()
            all_num = all_num + len(labels)
    test_acc = accuracy_num / all_num
    print("test accuracy:{}".format(test_acc))
    if test_acc > best_acc:
        filepath = os.path.join(checkpoint_path, 'checkpoint_best_model.pth')
        torch.save(model, filepath)
        best_acc = test_acc
