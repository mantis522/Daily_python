import torch
USE_GPU = torch.cuda.is_available()
print('USE_GPU={}'.format(USE_GPU))
if USE_GPU:
    print('current_device={}'.format(torch.cuda.current_device()))

from torchtext import data, vocab

SAMPLE_DATA_PATH = r"D:\ruin\data\summarization\summary\csv"

tokenizer = data.get_tokenizer('spacy')
TEXT = data.Field(tokenize=tokenizer, lower=True, eos_token='_eos_')

trn_data_fields = [("source", TEXT),
                   ("target", TEXT)]

trn, vld = data.TabularDataset.splits(path=f'{SAMPLE_DATA_PATH}',
                                     train=r"sample_train_ds.csv", validation=r"sample_valid_ds.csv",
                                     format='csv', skip_header=True, fields=trn_data_fields)

pre_trained_vector_type = 'glove.6B.200d'
TEXT.build_vocab(trn, vectors=pre_trained_vector_type )

print(TEXT.vocab.freqs.most_common(10))

batch_size = 64

train_iter, val_iter = data.BucketIterator.splits(
                        (trn, vld), batch_sizes=(batch_size, int(batch_size*1.6)),
                        device=(0 if USE_GPU else -1),
                        sort_key=lambda x: len(x.source),
                        shuffle=True, sort_within_batch=False, repeat=False)


class BatchTuple():
    def __init__(self, dataset, x_var, y_var):
        self.dataset, self.x_var, self.y_var = dataset, x_var, y_var

    def __iter__(self):
        for batch in self.dataset:
            x = getattr(batch, self.x_var)
            y = getattr(batch, self.y_var)
            yield (x, y)

    def __len__(self):
        return len(self.dataset)

train_iter_tuple = BatchTuple(train_iter, "source", "target")
val_iter_tuple = BatchTuple(val_iter, "source", "target")

# from fastai.text import ModelData
#
# model_data = ModelData(SAMPLE_DATA_PATH, trn_dl=train_iter_tuple, val_dl=val_iter_tuple)
#
# print(len(model_data.trn_dl), len(model_data.val_dl), len(TEXT.vocab))
