import re
import string
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator, Example, Dataset, Iterator
from torch.utils.data import DataLoader, random_split
import spacy
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
# from nltk.stem import WordNetLemmatizer
import random
import math
import time


class News_Dataset(Dataset):

    def __init__(self, path, fields, **kwargs):

        # path of directory containing inputs
        self.path = path
        # initialize fileds
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        # read Articles and summaries into pandas dataframe
        self.news_list = self._read_data()
        # load articles as torch text examples
        # I am not doing text pre-processing although I have written code for that
        examples = [Example.fromlist(list(item), fields) for item in self.news_list]
        # initialize
        super().__init__(examples, fields, **kwargs)

    def __len__(self):
        # return length of examples
        try:
            return len(self.examples)
        except TypeError:
            return 2 ** 32

    def __getitem__(self, index):
        # get items from examples
        return self.examples[index]

    def __iter__(self):
        # iterator for batch processing
        for x in self.examples:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)

    # function to read text files into pandas data frame
    def _read_data(self):
        # initialize variables
        Articles = []
        Summaries = []

        # loop over all files and read them into lists
        for d, path, filenames in tqdm(os.walk(self.path)):
            for file in filenames:
                if os.path.isfile(d + '/' + file):
                    if ('Summaries' in d + '/' + file):
                        with open(d + '/' + file, 'r', errors='ignore') as f:
                            summary = ' '.join([i.rstrip() for i in f.readlines()])
                            Summaries.append(summary)
                    else:
                        with open(d + '/' + file, 'r', errors='ignore') as f:
                            Article = ' '.join([i.rstrip() for i in f.readlines()])
                            Articles.append(Article)

        return zip(Articles, Summaries)

    # functions for pre-processing data
    # clean text data
    def _clean_data(self, text):
        # remove links
        text = self._remove_links(text)
        # remove numbers
        text = self._remove_numbers(text)
        # remove punctuations
        text = self._remove_punct(text)
        # word_list = self.tokenizer(text)
        # word_list = self._get_root(word_list)

        return text.lower()

    # remove punctuations
    def _remove_punct(self, text):
        nopunct = ''
        for c in text:
            if c not in string.punctuation:
                nopunct = nopunct + c
        return nopunct

    # remove numbers
    def _remove_numbers(self, text):
        return re.sub(r'[0-9]', '', text)

    # remove links
    def _remove_links(self, text):
        return re.sub(r'http\S+', '', text)

    # stemming
    def _get_root(self, word_list):
        ps = PorterStemmer()
        return [ps.stem(word) for word in word_list]

spacy_en = spacy.load('en')

def tokenize_en(text):
    # spacy tokenizer
    return [tok.text for tok in spacy_en.tokenizer(text)]

# fields for processing text data
# source field
SRC = Field(tokenize = tokenize_en,
            init_token = '<sos>',
            eos_token = '<eos>',
            fix_length= 500,
            lower = True)
# target field
TRG = Field(tokenize = tokenize_en,
            init_token = '<sos>',
            eos_token = '<eos>',
            fix_length= 200,
            lower = True)

news_data = News_Dataset(path=r"D:\ruin\data\summarization\BBC News Summary", fields=[SRC,TRG])

train_data, valid_data, test_data = news_data.split(split_ratio=[0.8,0.1,0.1], random_state=random.seed(21))

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)


def get_length_of_tokens(data):
    src = []
    trg = []
    for item in data.examples:
        src.append(len(vars(item)['src']))
        trg.append(len(vars(item)['trg']))

    return src, trg

src_len, trg_len = get_length_of_tokens(train_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 50

train_iterator, valid_iterator, test_iterator = Iterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device,
    sort_key= lambda x: len(x.src))


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        # initializations
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # we will use 2 layers for both encoder and decoder
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        # initialize
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # for decoder we will use n_directions 1
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        # fully connected layer to predict words
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, cell):
        # trg = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        trg = trg.unsqueeze(0)

        # trg = [1, batch size]

        embedded = self.dropout(self.embedding(trg))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size] where src_len is number of tokens in source sentence
        # trg = [trg len, batch size] same for trg_len
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim  # we don't have trg.shape[-1] here

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        dec_input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(dec_input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = trg[t] if teacher_force else top1

        return outputs

# seq2seq model's config variables
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 2
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

# initialize seq2seq model
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device)


class Seq2Seq_trainer(object):
    def __init__(self, model, train_iterator, valid_iterator, pad_index, device, clip, learning_rate):
        # initialize config variables
        self.model = model.to(device)
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.clip = clip
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
        self.model.apply(self.init_weights)
        print(f'The model has {self.count_parameters(self.model):,} trainable parameters')

    def init_weights(self, m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train(self):

        self.model.train()

        epoch_loss = 0

        for i, batch in enumerate(self.train_iterator):
            src = batch.src
            trg = batch.trg

            self.optimizer.zero_grad()

            output = self.model(src, trg)

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = self.criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(self.train_iterator)

    def evaluate(self, iterator):

        self.model.eval()

        epoch_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg

                output = self.model(src, trg, 0)  # turn off teacher forcing

                # trg = [trg len, batch size]
                # output = [trg len, batch size, output dim]

                output_dim = output.shape[-1]

                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

                # trg = [(trg len - 1) * batch size]
                # output = [(trg len - 1) * batch size, output dim]

                loss = self.criterion(output, trg)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def fit(self, nepochs):
        best_valid_loss = float('inf')

        for epoch in tqdm(range(nepochs)):

            start_time = time.time()

            train_loss = self.train()
            valid_loss = self.evaluate(self.valid_iterator)

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # torch.save(model.state_dict(), 'tut1-model.pt')
                print(f'Epoch with best validation loss: {epoch + 1:02}')

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    def predict(self, iterator):
        self.model.eval()

        with torch.no_grad():

            for i, batch in enumerate(tqdm(iterator)):

                src = batch.src
                trg = batch.trg

                output = self.model(src, trg, 0)  # turn off teacher forcing

                # trg = [trg len, batch size]
                # output = [trg len, batch size, output dim]

                if i == 0:
                    outputs = torch.argmax(output, -1)
                else:
                    outputs = torch.cat((outputs, torch.argmax(output, -1)), -1)

                # outputs = [trg_len, len(iterator)]
        return torch.transpose(outputs, 0, 1)

# config vaiables
pad_index = TRG.vocab.stoi[TRG.pad_token]
# initialize trainer
trainer = Seq2Seq_trainer(model, train_iterator, valid_iterator, pad_index, device, 1, 1e-3)
trainer.fit(30)

# evaluate on test data
test_loss = trainer.evaluate(test_iterator)
print(f'\t Test. Loss: {test_loss:.3f} |  Test. PPL: {math.exp(test_loss):7.3f}')