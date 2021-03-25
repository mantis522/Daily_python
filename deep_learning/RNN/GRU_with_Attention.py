import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import random
import pandas as pd
import re
from torchtext.data import TabularDataset
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

## RuntimeError: Input, output and indices must be on the current device 발생

SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)

# 하이퍼파라미터
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

 # torchtext.data 임포트

# 필드 정의
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=False)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

train_data, test_data = TabularDataset.splits(
        path=r"D:\ruin\data\csv_file\imdb_split", train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('테스트 샘플의 개수 : {}'.format(len(test_data)))

TEXT.build_vocab(train_data, min_freq=5) # 단어 집합 생성
LABEL.build_vocab(train_data)

n_classes = 1

vocab_size = len(TEXT.vocab)

train_data, val_data = train_data.split(split_ratio=0.8)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), sort=False,batch_size=BATCH_SIZE,
        shuffle=True, repeat=False)


class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.use_att = True
        self.att = Attention(hidden_dim)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)  # 첫번째 히든 스테이트를 0벡터로 초기화
        # h_0 = self._init_state(batch_size=x.size(0)) # 첫번째 히든 스테이트를 0벡터로 초기화
        output, hidden = self.gru(x, h_0)
        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        h_t = output[-1, :, :]  # (배치 크기, 은닉 상태의 크기(입력벡터 차원수))의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.
        ## [:, -1, :]는 첫번째는 미니배치 크기, 두번째는 마지막 은닉상태, 세번째는 모든 hidden_dim을 의미하는 것
        hidden = self.att(output, h_t)
        hidden = self.dropout(hidden)
        logit = self.out(hidden)  # (배치 크기, 은닉 상태의 크기) -> (배치 크기, 출력층의 크기)
        return logit


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, input, z):
        print(input.size())
        # input = [seq_len, batch_size, hidden dim]
        w1h = self.w1(input).transpose(0, 1)
        w2h = self.w2(z).unsqueeze(1)
        # w2h.size = [batch size, 1, hidden_dim]
        u_score = torch.tanh(w1h + w2h)
        u_score = self.V(u_score)
        att = torch.softmax(u_score, dim=1).transpose(1, 2)

        return torch.bmm(att, input.transpose(0, 1)).unsqueeze(1)

model = GRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in tqdm(iterator):
        optimizer.zero_grad()

        predictions = model(batch.review).squeeze(1)

        loss = criterion(predictions, batch.sentiment)

        acc = binary_accuracy(predictions, batch.sentiment)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.review).squeeze(1)

            loss = criterion(predictions, batch.sentiment)

            acc = binary_accuracy(predictions, batch.sentiment)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, val_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

# testing
model.load_state_dict(torch.load('tut1-model.pt'))

test_loss, test_acc = evaluate(model, test_iter, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')