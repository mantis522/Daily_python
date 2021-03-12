import pandas as pd
import torch
import chardet
from sklearn.model_selection import train_test_split
from torchtext import data
from torchtext.data import TabularDataset

# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU
    return device

def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True,
                  fix_length=200)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

glove = pd.read_csv(r"D:\ruin\data\glove.6B\glove.6B.100d.txt", sep=" ", quoting=3, header=None, index_col=0)
glove = df_to_tensor(glove)

train_data, test_data = TabularDataset.splits(
        path='D:/ruin/data/test/', train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

TEXT.build_vocab(train_data, min_freq=5, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

