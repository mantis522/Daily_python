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

BATCH_SIZE = 100 # バッチサイズ
EMBEDDING_DIM = 100 # 単語の埋め込み次元数
LSTM_DIM = 128 # LSTMの隠れ層の次元数
VOCAB_SIZE =TEXT.vocab.vectors.size()[0] # 全単語数
TAG_SIZE = 2 # 今回はネガポジ判定を行うのでネットワークの最後のサイズは2
DA = 64 # AttentionをNeural Networkで計算する際の重み行列のサイズ
R = 3 # Attentionを３層重ねて見る

class BiLSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, vocab_size):
        super(BiLSTMEncoder, self).__init__()
        self.lstm_dim = lstm_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 学習済単語ベクトルをembeddingとしてセットする
        self.word_embeddings.weight.data.copy_(TEXT.vocab.vectors)

        # 単語ベクトルを誤差逆伝播で更新させないためにrequires_gradをFalseに設定する
        self.word_embeddings.requires_grad_ = False

        # bidirectional=Trueでお手軽に双方向のLSTMにできる
        self.bilstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True, bidirectional=True)

    def forward(self, text):
        embeds = self.word_embeddings(text)

        # 各隠れ層のベクトルがほしいので第１戻り値を受け取る
        out, _ = self.bilstm(embeds)

        # 前方向と後ろ方向の各隠れ層のベクトルを結合したままの状態で返す
        return out

class SelfAttention(nn.Module):
  def __init__(self, lstm_dim, da, r):
    super(SelfAttention, self).__init__()
    self.lstm_dim = lstm_dim
    self.da = da
    self.r = r
    self.main = nn.Sequential(
        # Bidirectionalなので各隠れ層のベクトルの次元は２倍のサイズになってます。
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

encoder = BiLSTMEncoder(EMBEDDING_DIM, LSTM_DIM, VOCAB_SIZE).to(device)
classifier = SelfAttentionClassifier(LSTM_DIM, DA, R, TAG_SIZE).to(device)
loss_function = nn.NLLLoss()

# 複数のモデルを from itertools import chain で囲えばoptimizerをまとめて1つにできる
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

def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)

def mk_html(sentence, attns):
    html = ""
    for word, attn in zip(sentence, attns):
        html += ' ' + highlight(
            TEXT.vocab.itos[word],
            attn
        )
    return html


id2ans = {'0': 'positive', '1':'negative'}

_, test_iter = data.Iterator.splits((train, test), batch_sizes=(1, 1), device=device, repeat=False, sort=False)

n = random.randrange(len(test_iter))

for batch in itertools.islice(test_iter, n-1,n):
    x = batch.review[0]
    y = batch.sentiment
    encoder_outputs = encoder(x)
    output, attn = classifier(encoder_outputs)
    pred = output.data.max(1, keepdim=True)[1]

    display(HTML('【正解】' + id2ans[str(y.item())] + '\t【予測】' + id2ans[str(pred.item())] + '<br><br>'))
    for i in range(attn.size()[2]):
      display(HTML(mk_html(x.data[0], attn.data[0,:,i]) + '<br><br>'))