import pandas as pd
import torch
import chardet
from sklearn.model_selection import train_test_split
from torchtext import data
from torchtext.data import TabularDataset, BucketIterator
import torch.nn as nn
import torch.nn.functional as F
import os

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True,
                  fix_length=200)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

train_data, test_data = TabularDataset.splits(
        path='D:/ruin/data/test/', train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

TEXT.build_vocab(train_data, min_freq=5, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

train_data, val_data = train_data.split(split_ratio=0.8)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), sort=False,batch_size=64,
        shuffle=True, repeat=False)

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim)  # convert sparse 1 hot encoded vectors to embeddings (glove embedding will be used here)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers,
                          bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        ## [:, -1, :]는 첫번째는 미니배치 크기, 두번째는 마지막 은닉상태, 세번째는 모든 hidden_dim을 의미하는 것
        ## 양방향은 미니배치로 조정해야 하나?
        return self.fc(hidden.squeeze(0))

epochs = 10
input_dim = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 20
output_dim = 1
n_layers = 2 # for mutil layer rnn
bidirectional = True
dropout = 0.5
lr = 0.0001

model = BiRNN(input_dim,
            embedding_dim,
            hidden_dim,
            output_dim,
            n_layers,
            bidirectional,
            dropout).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train(model, optimizer, train_iter):
    model.train() ## model 학습
    for b, batch in enumerate(train_iter):
        x, y = batch.review.to(device), batch.sentiment.to(device) ## 각각 x, y에 리뷰 데이터랑 라벨 넣기
        # y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()  # 갱신할 변수들에 대한 모든 변화도를 0으로 만듭니다. 이렇게 하는 이유는
        # 기본적으로 .backward()를 호출할 때마다 변화도가 버퍼(buffer)에 (덮어쓰지 않고) 누적되기 때문입니다.

        logit = model(x) ## 모델에 x를 넣고
        loss = F.cross_entropy(logit, y) ## 크로스 엔트로피 함수로 loss 함수 구함
        loss.backward() ## 역전파로 계산하면서
        optimizer.step() ## 매개변수 업데이트

def evaluate(model, val_iter):
    """evaluate model"""
    model.eval() ## eval은 그냥 함순가 봄
    corrects, total_loss = 0, 0 ## total_loss는 대충 알겠는데 corrects는 뭔지 모르겠네
    for batch in val_iter:
        x, y = batch.review.to(device), batch.sentiment.to(device)
        # y.data.sub_(1) # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy

best_val_loss = None
for e in range(1, epochs+1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss

model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter)
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))