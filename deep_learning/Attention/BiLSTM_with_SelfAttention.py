import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import random
from torchtext.data import TabularDataset
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vectors
from itertools import chain
from sklearn.metrics import classification_report
import itertools
from IPython.display import display, HTML

# 필드 정의
TEXT = data.Field(sequential=True, tokenize=str.split, lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)

train, test = TabularDataset.splits(
        path=r"D:\ruin\data\csv_file\imdb_split", train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

glove_vectors = Vectors(r"D:\ruin\data\glove.6B\glove.6B.100d.txt")
TEXT.build_vocab(train, vectors=glove_vectors, min_freq=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 80
EMBEDDING_DIM = 100
LSTM_DIM = 128
VOCAB_SIZE =TEXT.vocab.vectors.size()[0]
TAG_SIZE = 2
DA = 64
R = 3 ## attention 3층으로?

class BiLSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, vocab_size):
        super(BiLSTMEncoder, self).__init__()
        self.lstm_dim = lstm_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.word_embeddings.weight.data.copy_(TEXT.vocab.vectors)

        self.word_embeddings.requires_grad_ = False

        self.bilstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True, bidirectional=True)

    def forward(self, text):
        embeds = self.word_embeddings(text)

        out, _ = self.bilstm(embeds)

        return out

## a = softmax(WS2 x tanh(Ws1 H^T))
## Ws1은 weight matrix Da X 2u 크기
## WS2는 r x da 크기

class SelfAttention(nn.Module):
    def __init__(self, lstm_dim, da, r):
        super(SelfAttention, self).__init__()
        self.lstm_dim = lstm_dim
        self.da = da
        self.r = r
        self.main = nn.Sequential(
            ## H는 lstm hidden dim 합이니까 bi-lstm이라 2곱하고
            nn.Linear(lstm_dim * 2, da),
            ## 앞에오는게 input 뒤에 오는게 output
            nn.Tanh(),
            nn.Linear(da, r)
            ## WS2는
        )
    def forward(self, out):
        return F.softmax(self.main(out), dim=1)

class SelfAttentionClassifier(nn.Module):
    def __init__(self, lstm_dim, da, r, tagset_size):
        super(SelfAttentionClassifier, self).__init__()
        self.lstm_dim = lstm_dim
        self.r = r
        self.attn = SelfAttention(lstm_dim, da, r)
        self.main = nn.Linear(lstm_dim * 6, tagset_size)
        ## bidirectional 2에 3개 concat 해서 6 곱하는 것

    def forward(self, out):
        attention_weight = self.attn(out)
        ## out = [batch size, seq_len, LSTM_DIM * 2]
        ## attention_weight = [batch size, seq_len, R]
        m1 = (out * attention_weight[:,:,0].unsqueeze(2)).sum(dim=1)
        ## m1 = [batch size, LSTM DIM * 2]
        m2 = (out * attention_weight[:,:,1].unsqueeze(2)).sum(dim=1)
        m3 = (out * attention_weight[:,:,2].unsqueeze(2)).sum(dim=1)
        feats = torch.cat([m1, m2, m3], dim=1)
        ## feat = [batch size, LSTM DIM * 6]
        return F.log_softmax(self.main(feats)), attention_weight

encoder = BiLSTMEncoder(EMBEDDING_DIM, LSTM_DIM, VOCAB_SIZE).to(device)
classifier = SelfAttentionClassifier(LSTM_DIM, DA, R, TAG_SIZE).to(device)
loss_function = nn.NLLLoss()

optimizer = torch.optim.Adam(chain(encoder.parameters(), classifier.parameters()), lr=0.001)

train_iter, test_iter = data.Iterator.splits((train, test), batch_sizes=(BATCH_SIZE, BATCH_SIZE), device=device, repeat=False, sort=False)

losses = []
for epoch in range(10):
    all_loss = 0

    for idx, batch in enumerate(train_iter):
        batch_loss = 0
        encoder.zero_grad()
        classifier.zero_grad()

        text_tensor = batch.review[0]
        label_tensor = batch.sentiment
        out = encoder(text_tensor)
        score, attn = classifier(out)
        batch_loss = loss_function(score, label_tensor)
        batch_loss.backward()
        optimizer.step()
        all_loss += batch_loss.item()
    print("epoch", epoch, "\t" , "loss", all_loss)

answer = []
prediction = []
with torch.no_grad():
    for batch in test_iter:

        text_tensor = batch.review[0]
        label_tensor = batch.sentiment

        out = encoder(text_tensor)
        score, _ = classifier(out)
        _, pred = torch.max(score, 1)

        prediction += list(pred.cpu().numpy())
        answer += list(label_tensor.cpu().numpy())
print(classification_report(prediction, answer, target_names=['positive', 'negative']))