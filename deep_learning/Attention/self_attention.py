import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import random
from torchtext.data import TabularDataset
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vectors
from itertools import chain
from tqdm import tqdm
from sklearn.metrics import classification_report
import itertools

TEXT = data.Field(sequential=True, tokenize=str.split, lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)

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

class SelfAttention(nn.Module):
    def __init__(self, lstm_dim, da, r):
        super(SelfAttention, self).__init__()
        self.lstm_dim = lstm_dim
        self.da = da
        self.r = r
        self.main = nn.Sequential(
        nn.Linear(lstm_dim * 2, da),
        nn.Tanh(),
        nn.Linear(da, r)
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

    def forward(self, out):
        attention_weight = self.attn(out)
        m1 = (out * attention_weight[:,:,0].unsqueeze(2)).sum(dim=1)
        m2 = (out * attention_weight[:,:,1].unsqueeze(2)).sum(dim=1)
        m3 = (out * attention_weight[:,:,2].unsqueeze(2)).sum(dim=1)
        feats = torch.cat([m1, m2, m3], dim=1)
        return F.log_softmax(self.main(feats)), attention_weight

def train(encoder, classifier, iterator, optimizer, criterion):
    losses = []
    encoder.train()
    all_loss = 0

    for idx, batch in enumerate(iterator):
        batch_loss = 0
        encoder.zero_grad()
        classifier.zero_grad()

        text_tensor = batch.review[0]
        label_tensor = batch.sentiment
        out = encoder(text_tensor)
        score, attn = classifier(out)
        batch_loss = criterion(score, label_tensor)
        batch_loss.backward()
        optimizer.step()

def evalutate(encoder, classifier, iterator):
    answer = []
    prediction = []
    with torch.no_grad():
        for batch in iterator:
            text_tensor = batch.review[0]
            label_tensor = batch.sentiment

            out = encoder(text_tensor)
            score, _ = classifier(out)
            _, pred = torch.max(score, 1)

            prediction += list(pred.cpu().numpy())
            answer += list(label_tensor.cpu().numpy())

def main():
    train, test = TabularDataset.splits(
        path=r"D:\ruin\data\csv_file\imdb_split", train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

    glove_vectors = Vectors(r"D:\ruin\data\glove.6B\glove.6B.100d.txt")
    TEXT.build_vocab(train, vectors=glove_vectors, min_freq=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 100  # バッチサイズ
    EMBEDDING_DIM = 100  # 単語の埋め込み次元数
    LSTM_DIM = 128  # LSTMの隠れ層の次元数
    VOCAB_SIZE = TEXT.vocab.vectors.size()[0]  # 全単語数
    TAG_SIZE = 2  # 今回はネガポジ判定を行うのでネットワークの最後のサイズは2
    DA = 64  # AttentionをNeural Networkで計算する際の重み行列のサイズ
    R = 3  # Attentionを３層重ねて見る

    encoder = BiLSTMEncoder(EMBEDDING_DIM, LSTM_DIM, VOCAB_SIZE).to(device)
    classifier = SelfAttentionClassifier(LSTM_DIM, DA, R, TAG_SIZE).to(device)
    loss_function = nn.NLLLoss()

    optimizer = torch.optim.Adam(chain(encoder.parameters(), classifier.parameters()), lr=0.001)

    train_iter, test_iter = data.Iterator.splits((train, test), batch_sizes=(BATCH_SIZE, BATCH_SIZE),
                                                 device=device, repeat=False, sort=False)

