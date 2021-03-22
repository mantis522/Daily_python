import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import random
import pandas as pd
import re
from torchtext.data import TabularDataset
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

## https://github.com/yjfiejd/2019xuexi/blob/cc4ab107568eb02e65767b405133d1a720fc73cf/Pytorch_new/%E7%AC%AC%E5%9B%9B%E8%AF%BE.py 참조

# 하이퍼파라미터
RANDOM_SEED = 2020
MAX_VOCAB_SIZE = 25000
BATCH_SIZE = 128

torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

 # torchtext.data 임포트

# 필드 정의
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True)

LABEL = data.LabelField(sequential=False,
                   use_vocab=False,
                   batch_first=True,
                   is_target=True, dtype=torch.long)

class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(DEVICE) # 첫번째 히든 스테이트를 0벡터로 초기화
        # h_0 = self._init_state(batch_size=x.size(0)) # 첫번째 히든 스테이트를 0벡터로 초기화
        x, _ = self.gru(x, h_0)  # GRU의 리턴값은 (배치 크기, 시퀀스 길이, 은닉 상태의 크기)
        h_t = x[:,-1,:] # (배치 크기, 은닉 상태의 크기(입력벡터 차원수))의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.
        ## [:, -1, :]는 첫번째는 미니배치 크기, 두번째는 마지막 은닉상태, 세번째는 모든 hidden_dim을 의미하는 것
        self.dropout(h_t)
        logit = self.out(h_t)  # (배치 크기, 은닉 상태의 크기) -> (배치 크기, 출력층의 크기)
        return logit

    # def _init_state(self, batch_size=1):
    #     weight = next(self.parameters()).data
    #     return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

def binary_accuracy(pred, target, threshold=0.5):
    preds = torch.sigmoid(pred) > threshold
    print(preds.size())
    print(preds)
    print(target.size())
    print(target)
    correct = (preds==target).float()
    return correct.mean()

def train(model, data_loader, optimizer, criterion):
    model.train()

    epoch_loss = []
    epoch_acc = []

    pbar = tqdm(data_loader)
    for data in pbar:
        text = data.review.to(DEVICE)
        print(text.size())
        # text_length = data.review[1]
        label = data.sentiment.to(DEVICE)

        pred = model(text)
        loss = criterion(pred, label.unsqueeze(dim=1))
        # grad clearing
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_acc = binary_accuracy(pred, label.unsqueeze(dim=1)).item()

        epoch_loss.append(batch_loss)
        epoch_acc.append(batch_acc)

        pbar.set_description('train => acc {} loss {}'.format(batch_acc, batch_loss))

    return sum(epoch_acc) / len(data_loader), sum(epoch_loss) / len(data_loader)


def test(model, data_loader, criterion):
    model.eval()

    epoch_loss = []
    epoch_acc = []
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for data in pbar:
            text = data.review.to(DEVICE)
            label = data.sentiment.to(DEVICE)
            pred = model(text)
            loss = criterion(pred, label.unsqueeze(dim=1))

            batch_loss = loss.item()
            batch_acc = binary_accuracy(pred, label.unsqueeze(dim=1)).item()

            epoch_loss.append(batch_loss)
            epoch_acc.append(batch_acc)

            pbar.set_description('eval() => acc {} loss {}'.format(batch_acc, batch_loss))

    return sum(epoch_acc) / len(data_loader), sum(epoch_loss) / len(data_loader)

def main():
    train_data, test_data = TabularDataset.splits(
        path=r"D:\ruin\data\csv_file\imdb_split", train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

    train_data, eval_data = train_data.split(random_state=random.seed(RANDOM_SEED))

    print('Number of train data {}'.format(len(train_data)))
    print('Number of val data {}'.format(len(eval_data)))
    print('Number of test data {}'.format(len(test_data)))

    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",
                     min_freq=10)
    LABEL.build_vocab(train_data)

    print('Unique token in Text vocabulary {}'.format(len(TEXT.vocab)))  # 250002(<unk>, <pad>)
    print(TEXT.vocab.itos)
    print('Unique token in LABEL vocabulary {}'.format(len(LABEL.vocab)))
    print(LABEL.vocab.itos)

    print('Done')

    train_iter = data.BucketIterator(train_data, batch_size=BATCH_SIZE, device=DEVICE, shuffle=True)
    eval_iter, test_iter = data.BucketIterator.splits((eval_data, test_data), batch_size=BATCH_SIZE, device=DEVICE,
                                                      sort_key=lambda x: len(x.review),
                                                      sort_within_batch=True)

    for batch_data in train_iter:
        print(batch_data.review)  # text, text_length
        print(batch_data.sentiment)  # label
        break

    vocab_size = len(TEXT.vocab)
    n_classes = 2

    model = GRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    model = model.to(DEVICE)
    EPOCH = 1

    best_eval_loss = float('inf')

    for epoch in range(EPOCH):
        print('{}/{}'.format(epoch, EPOCH))
        train_acc, train_loss = train(model, train_iter, optimizer=optimizer, criterion=criterion)
        eval_acc, eval_loss = test(model, eval_iter, criterion=criterion)

        print('Train => acc {:.3f}, loss {:4f}'.format(train_acc, train_loss))
        print('Eval => acc {:.3f}, loss {:4f}'.format(eval_acc, eval_loss))
        scheduler.step()

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss

    test_acc, test_loss = test(model, test_iter, criterion=criterion)
    print('Test Eval => acc {:.3f}, loss {:4f}'.format(test_acc, test_loss))

if __name__ == "__main__":
    main()