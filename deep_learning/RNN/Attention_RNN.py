import torch
from torchtext import data
import random
from torchtext.data import TabularDataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 1234

TEXT = data.Field(tokenize = str.split)
LABEL = data.LabelField(dtype = torch.float)

train_data, test_data = TabularDataset.splits(
            path=r"D:\ruin\data\csv_file\imdb_split", train='train_data.csv', test='test_data.csv', format='csv',
            fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

print('Number of train data {}'.format(len(train_data)))
print('Number of val data {}'.format(len(valid_data)))
print('Number of test data {}'.format(len(test_data)))

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data,
                     max_size = MAX_VOCAB_SIZE,
                     vectors = 'glove.6B.100d',
                     unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)

BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = BATCH_SIZE, sort_key=lambda x: len(x.review),
        device = device)

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.use_att = True
        self.att = Attention(hidden_dim)

    def forward(self, text):
        # text = [sent len, batch size]
        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]
        output, hidden = self.rnn(embedded)

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        h_t = hidden.view(text.shape[1], -1)
        hidden = self.att(output, h_t)
        pred = self.fc(hidden.squeeze(0)).view(-1, 1)

        return pred

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, input, z):
        w1h = self.w1(input).transpose(0, 1)
        w2h = self.w2(z).unsqueeze(1)
        u_score = torch.tanh(w1h + w2h)
        u_score = self.V(u_score)
        att = torch.softmax(u_score, dim=1).transpose(1, 2)

        return torch.bmm(att, input.transpose(0, 1)).unsqueeze(1)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)

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

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

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

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')