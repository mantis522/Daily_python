import os
import random
import torch
import torch.nn  as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchtext.data import TabularDataset
from torchtext import data
from tqdm import tqdm
import shutil

device = 'cuda' if torch.cuda.is_available() else 'cpu'

RANDOM_SEED = 2020
MAX_VOCAB_SIZE = 25000  ## 상위 25k개 단어만 사전에 넣겠다는 의미.
BATCH_SIZE = 128
SEED = 1234

torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize=str.split, include_lengths=True)
LABEL = data.LabelField(sequential=False, dtype=torch.float32)

train_data, test_data = TabularDataset.splits(
            path=r"D:\ruin\data\csv_file\imdb_split", train='train_data.csv', test='test_data.csv', format='csv',
            fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

print('Number of train data {}'.format(len(train_data)))
print('Number of val data {}'.format(len(valid_data)))
print('Number of test data {}'.format(len(test_data)))

TEXT.build_vocab(train_data,
                     max_size = MAX_VOCAB_SIZE,
                     vectors = 'glove.6B.100d',
                     unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = BATCH_SIZE, sort_key=lambda x: len(x.review),
        device = device)

class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layer, pad_index,
                 bidirectional=False, dropout=0.5):
        super(BiLSTMSentiment, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, ## size of the dictionary of embeddings
                                      embedding_dim=embedding_dim, ## the size of each embedding vector
                                      padding_idx=pad_index)
        # num_embeddings : 임베딩을 할 단어들의 개수. 다시 말해 단어 집합의 크기입니다.
        # embedding_dim : 임베딩 할 벡터의 차원입니다. 사용자가 정해주는 하이퍼파라미터입니다.
        # padding_idx : 선택적으로 사용하는 인자입니다. 패딩을 위한 토큰의 인덱스를 알려줍니다.

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layer,
                            bidirectional=bidirectional,
                            dropout=dropout)

        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.use_att = True
        self.att = Attention(hidden_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, text, text_length):
        """
        :param text: (seq_len, batch_size)
        :param text_length:
        :return:
        """
        # embedded => [seq_len, batch_size, embedding_dim]
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        text_length = text_length.cpu()  # compatible torch=1.7.0
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_length, batch_first=False, enforce_sorted=False)
        ## pad_packed_sequence에 대해서 : https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html
        # lstm
        # h_n => (num_direction * num_layers, batch_size, hidden_size)
        # c_n => (num_direction * num_layers, batch_size, hidden_size)
        packed_output, (h_n, c_n) = self.lstm(packed_embedded)

        # unpacked sequence
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=False)

        # hidden => (batch_size, hidden_size*num_direction)
        # only use hidden state of last layer
        hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        hidden = self.att(output, hidden)

        hidden = self.dropout(hidden)

        out = self.fc(hidden)
        return (out)

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, input, z):
        w1h = self.w1(input).transpose(0, 1)
        # 위의 output과 동일한 크기
        w2h = self.w2(z).unsqueeze(1)
        # 위의 hidden과 동일한 크기
        u_score = torch.tanh(w1h + w2h)
        u_score = self.V(u_score)
        att = torch.softmax(u_score, dim=1).transpose(1, 2)

        return torch.bmm(att, input.transpose(0, 1)).unsqueeze(1)

def binary_accuracy(pred, target, threshold=0.5):
    preds = torch.sigmoid(pred) > threshold
    correct = (preds==target).float()
    return correct.mean()

def train(model, data_loader, optimizer, criterion):
    model.train()

    epoch_loss = []
    epoch_acc = []

    pbar = tqdm(data_loader)
    for data in pbar:
        text = data.review[0].to(device)
        text_length = data.review[1]
        label = data.sentiment.to(device)

        pred = model(text, text_length)
        loss = criterion(pred, label.unsqueeze(dim=1))
        # grad clearing
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_acc = binary_accuracy(pred, label.unsqueeze(dim=1)).item()
        batch_size = text.shape[1]

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
            text = data.review[0].to(device)
            text_length = data.review[1]
            label = data.sentiment.to(device)
            pred = model(text, text_length)
            loss = criterion(pred, label.unsqueeze(dim=1))

            batch_loss = loss.item()
            batch_acc = binary_accuracy(pred, label.unsqueeze(dim=1)).item()

            epoch_loss.append(batch_loss)
            epoch_acc.append(batch_acc)

            pbar.set_description('eval() => acc {} loss {}'.format(batch_acc, batch_loss))

    return sum(epoch_acc) / len(data_loader), sum(epoch_loss) / len(data_loader)

VOCAB_SIZE = len(TEXT.vocab)
HIDDEN_SIZE = 256
OUTPUT_SIZE = 1
NUM_LAYER = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
EMBEDDING_DIM = 100
PAD_INDEX = TEXT.vocab.stoi[TEXT.pad_token]
UNK_INDEX = TEXT.vocab.stoi[TEXT.unk_token]

model = BiLSTMSentiment(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE,
                        output_size=OUTPUT_SIZE, num_layer=NUM_LAYER, bidirectional=BIDIRECTIONAL,
                        dropout=DROPOUT, pad_index=PAD_INDEX)

pretrained_embedding = TEXT.vocab.vectors
pretrained_embedding[PAD_INDEX] = 0
pretrained_embedding[UNK_INDEX] = 0

model.embedding.weight.data.copy_(pretrained_embedding)

    # optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # criterion
criterion = nn.BCEWithLogitsLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

model = model.to(device)
EPOCH = 1
# MODEL_PATH = './output/bilstm_model.pth'
# BEST_MODEL_PATH = './output/bilstm_model_best.pth'
best_eval_loss = float('inf')

for epoch in range(EPOCH):
    print('{}/{}'.format(epoch, EPOCH))
    train_acc, train_loss = train(model, train_iterator, optimizer=optimizer, criterion=criterion)
    eval_acc, eval_loss = test(model, valid_iterator, criterion=criterion)

    print('Train => acc {:.3f}, loss {:4f}'.format(train_acc, train_loss))
    print('Eval => acc {:.3f}, loss {:4f}'.format(eval_acc, eval_loss))
    scheduler.step()

    # save model
    state = {
        'vocab_size': VOCAB_SIZE,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_size': HIDDEN_SIZE,
        'output_size': OUTPUT_SIZE,
        'num_layer': NUM_LAYER,
        'bidirectional': BIDIRECTIONAL,
        'dropout': DROPOUT,
        'state_dict': model.state_dict(),
        'pad_index': PAD_INDEX,
        'unk_index': UNK_INDEX,
        'text_vocab': TEXT.vocab.stoi,
    }

    # os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    # torch.save(state, MODEL_PATH)
    if eval_loss < best_eval_loss:
        # shutil.copy(MODEL_PATH, BEST_MODEL_PATH)
        best_eval_loss = eval_loss