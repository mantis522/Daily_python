import os
import random
import torch
import torch.nn  as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchtext.data import TabularDataset
from torchtext import data
from torchtext.data import Iterator
from tqdm import tqdm
import shutil

device = 'cuda' if torch.cuda.is_available() else 'cpu'

RANDOM_SEED = 2020
MAX_VOCAB_SIZE = 25000  ## 상위 25k개 단어만 사전에 넣겠다는 의미.
BATCH_SIZE = 128        ##

torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize=str.split, include_lengths=True, lower=True)
LABEL = data.LabelField(sequential=False, dtype=torch.float32)

train_data, test_data = TabularDataset.splits(
        path=r"D:\ruin\data\csv_file\imdb_split", train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

train_data, eval_data = train_data.split(random_state = random.seed(RANDOM_SEED))

print('Number of train data {}'.format(len(train_data)))
print('Number of val data {}'.format(len(eval_data)))
print('Number of test data {}'.format(len(test_data)))

print(vars(train_data[0]))

TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     min_freq=10)
LABEL.build_vocab(train_data)
batch_size = 10

train_loader = Iterator(dataset=train_data, batch_size = batch_size)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)

batch = next(iter(train_loader)) # 첫번째 미니배치

print(batch.review)

#
# train_iter = data.BucketIterator(train_data, batch_size=BATCH_SIZE, device=device, shuffle=True)
# eval_iter, test_iter = data.BucketIterator.splits((eval_data, test_data), batch_size=BATCH_SIZE, device=device,
#                                                       sort_key=lambda x: len(x.review),
#                                                       sort_within_batch=True)
#
# print(vars(test_iter))