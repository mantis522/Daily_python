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
    for epoch in range(10):
        all_loss = 0

        for idx, batch in enumerate(iterator):
            batch_loss = 0
            encoder.zero_grad()
            classifier.zero_grad()

            text_tensor = batch.review[0]
