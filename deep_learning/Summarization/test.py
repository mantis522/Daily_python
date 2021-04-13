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

