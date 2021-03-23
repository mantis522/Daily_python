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

TEXT = data.Field(tokenize = str.split,
                  batch_first = True)
LABEL = data.LabelField(dtype = torch.float, batch_first=True)

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[0], embedding_dim))
        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[1], embedding_dim))
        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[2], embedding_dim))
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    pbar = tqdm(iterator)

    for data in pbar:
        optimizer.zero_grad()
        predictions = model(data.review).squeeze(1)
        loss = criterion(predictions, data.sentiment)
        acc = binary_accuracy(predictions, data.sentiment)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        batch_acc = acc.item()
        epoch_loss += batch_loss
        epoch_acc += batch_acc

        pbar.set_description('train => acc {} loss {}'.format(batch_acc, batch_loss))

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        pbar = tqdm(iterator)
        for data in pbar:
            predictions = model(data.review).squeeze(1)
            loss = criterion(predictions, data.sentiment)
            acc = binary_accuracy(predictions, data.sentiment)
            batch_loss = loss.item()
            batch_acc = acc.item()
            epoch_loss += batch_loss
            epoch_acc += batch_acc

            pbar.set_description('eval() => acc {} loss {}'.format(batch_acc, batch_loss))

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

def main():
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

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [3,4,5]
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

    pretrained_embeddings = TEXT.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = torch.optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


    best_valid_loss = float('inf')
    N_EPOCHS = 5

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc * 100:.2f}%')

if __name__ == "__main__":
    main()
