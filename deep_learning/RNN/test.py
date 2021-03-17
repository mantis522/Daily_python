import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext import data

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize='spacy',
                  lower=True,
                  batch_first=True,
                  fix_length=200)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)