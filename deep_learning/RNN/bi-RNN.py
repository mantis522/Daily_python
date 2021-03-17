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
BATCH_SIZE = 128        ##

torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

# define datatype
# method 1
# python -m spacy download en
# method 2
# step 1: manual download from https://github-production-release-asset-2e65be.s3.amazonaws.com/84940268/69ded28e-c3ef-11e7-94dc-d5b03d9597d8?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20201214%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20201214T064457Z&X-Amz-Expires=300&X-Amz-Signature=631b41e8491a84dfb7c492f336d728f116a04f677c33cf709dd719d5cf4c126f&X-Amz-SignedHeaders=host&actor_id=26615837&key_id=0&repo_id=84940268&response-content-disposition=attachment%3B%20filename%3Den_core_web_sm-2.0.0.tar.gz&response-content-type=application%2Foctet-stream
# step 2: remove to /home/alex/anaconda3/envs/pytorch/lib/python3.6/site-packages/spacy/data
# step 3: $ pip install en_core_web_sm-2.0.0.tar.gz
# step 4: $ spacy link en_core_web_sm en

# TEXT = data.Field(tokenize='spacy', fix_length=1000)
TEXT = data.Field(tokenize='spacy', include_lengths=True)
LABEL = data.LabelField(sequential=False, dtype=torch.float32)

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

        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

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
        if self.lstm.bidirectional:
            hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)

        else:
            hidden = h_n[-1:, :, :]

        hidden = self.dropout(hidden)

        out = self.fc(hidden)
        return out


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


def main():
    # -----------------get train, val and test data--------------------
    train_data, test_data = TabularDataset.splits(
        path='D:/ruin/data/test/', train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

    train_data, eval_data = train_data.split(random_state = random.seed(RANDOM_SEED))

    print('Number of train data {}'.format(len(train_data)))
    print('Number of val data {}'.format(len(eval_data)))
    print('Number of test data {}'.format(len(test_data)))

    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",
                     min_freq=10)
    LABEL.build_vocab(train_data)
    print('Unique token in Text vocabulary {}'.format(len(TEXT.vocab)))  # 250002(<unk>, <pad>)
    print(TEXT.vocab.itos)
    print('Unique token in LABEL vocabulary {}'.format(len(LABEL.vocab)))
    print(TEXT.vocab.itos)

    print('Top 20 frequency of word: \n {}'.format(TEXT.vocab.freqs.most_common(20)))
    print('Embedding shape {}'.format(TEXT.vocab.vectors.size))

    print('Done')

    # generate dataloader
    train_iter = data.BucketIterator(train_data, batch_size=BATCH_SIZE, device=device, shuffle=True)
    eval_iter, test_iter = data.BucketIterator.splits((eval_data, test_data), batch_size=BATCH_SIZE, device=device,
                                                      sort_key=lambda x: len(x.review),
                                                      sort_within_batch=True)
    ## https://stackoverflow.com/questions/58241313/understanding-typeerror-not-supported-between-instances-of-example-and-e
    ## sort_key=lambda x: len(x.review) 추가에 대해선 위 링크에서 설명.
    for batch_data in train_iter:
        print(batch_data.review)  # text, text_length
        print(batch_data.sentiment)  # label
        break

    # construct model
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

    # load pretrained weight of embedding layer
    pretrained_embedding = TEXT.vocab.vectors
    print(pretrained_embedding)
    pretrained_embedding[PAD_INDEX] = 0
    pretrained_embedding[UNK_INDEX] = 0
    print(pretrained_embedding)

    model.embedding.weight.data.copy_(pretrained_embedding)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # criterion
    criterion = nn.BCEWithLogitsLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model = model.to(device)
    EPOCH = 1
    MODEL_PATH = './output/bilstm_model.pth'
    BEST_MODEL_PATH = './output/bilstm_model_best.pth'
    best_eval_loss = float('inf')
    for epoch in range(EPOCH):
        print('{}/{}'.format(epoch, EPOCH))
        train_acc, train_loss = train(model, train_iter, optimizer=optimizer, criterion=criterion)
        eval_acc, eval_loss = test(model, eval_iter, criterion=criterion)

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

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(state, MODEL_PATH)
        if eval_loss < best_eval_loss:
            shutil.copy(MODEL_PATH, BEST_MODEL_PATH)
            best_eval_loss = eval_loss

    test_acc, test_loss = test(model, test_iter, criterion=criterion)
    print('Test Eval => acc {:.3f}, loss {:4f}'.format(test_acc, test_loss))
if __name__ == "__main__":
    main()