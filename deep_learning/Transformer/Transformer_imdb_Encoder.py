import os
import random
import torch
import torch.nn  as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchtext.data import TabularDataset
from torchtext import data
from tqdm import tqdm
from torchtext.vocab import Vectors
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_imdb(batch_size, max_length):
    TEXT = data.Field(tokenize=str.split, include_lengths=True, batch_first=True, fix_length=max_length)
    LABEL = data.LabelField(sequential=False)

    train_data, test_data = TabularDataset.splits(
        path=r"D:\ruin\data\csv_file\imdb_split", train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

    glove_vectors = Vectors(r"D:\ruin\data\glove.6B\glove.6B.100d.txt")

    TEXT.build_vocab(train_data,
                     vectors=glove_vectors,
                     min_freq=10)
    LABEL.build_vocab(train_data)

    train_iter, test_iter = data.BucketIterator.splits((train_data, test_data), batch_size=batch_size, device=-1,
                                                      sort_key=lambda x: len(x.review))

    return train_iter, test_iter, TEXT.vocab.vectors, TEXT.vocab

def get_pos_onehot(length):
    # initial zero matrix [length, length]
    onehot = torch.zeros(length, length)

    idxs = torch.arange(length).long().view(-1, 1)

    onehot.scatter_(1, idxs, 1)
    return onehot


class MultiHeadAttention(nn.Module):
    """
        A multihead attention module,
        using scaled dot-product attention.
    """

    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.head_size = int(self.hidden_size / num_heads)

        self.q_linear = nn.Linear(self.input_size, self.hidden_size)
        self.k_linear = nn.Linear(self.input_size, self.hidden_size)
        self.v_linear = nn.Linear(self.input_size, self.hidden_size)
        #
        self.joint_linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        # project the queries, keys and values by their respective weight matrices
        q_proj = self.q_linear(q).view(q.size(0), q.size(1), self.num_heads, self.head_size).transpose(1, 2)
        k_proj = self.k_linear(k).view(k.size(0), k.size(1), self.num_heads, self.head_size).transpose(1, 2)
        v_proj = self.v_linear(v).view(v.size(0), v.size(1), self.num_heads, self.head_size).transpose(1, 2)

        # calculate attention weights
        unscaled_weights = torch.matmul(q_proj, k_proj.transpose(2, 3))
        weights = self.softmax(unscaled_weights / torch.sqrt(torch.Tensor([self.head_size * 1.0]).to(unscaled_weights)))

        # weight values by their corresponding attention weights
        weighted_v = torch.matmul(weights, v_proj)

        weighted_v = weighted_v.transpose(1, 2).contiguous()

        # do a linear projection of the weighted sums of values
        joint_proj = self.joint_linear(weighted_v.view(q.size(0), q.size(1), self.hidden_size))

        # store a reference to attention weights, for THIS forward pass,
        # for visualisation purposes
        self.weights = weights

        return joint_proj


class Block(nn.Module):
    """
        One block of the transformer.
        Contains a multihead attention sublayer
        followed by a feed forward network.
    """

    def __init__(self, input_size, hidden_size, num_heads, activation=nn.ReLU, dropout=None):
        super(Block, self).__init__()
        self.dropout = dropout

        self.attention = MultiHeadAttention(input_size, hidden_size, num_heads)
        self.attention_norm = nn.LayerNorm(input_size)

        ff_layers = [
            nn.Linear(input_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, input_size),
        ]

        if self.dropout:
            self.attention_dropout = nn.Dropout(dropout)
            ff_layers.append(nn.Dropout(dropout))
        # nn.Sequential(): Modules will be added to it in the order they are passed in the constructor. Alternatively,
        # an ordered dict of modules can also be passed in.
        self.ff = nn.Sequential(*ff_layers)
        self.ff_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        attended = self.attention_norm(self.attention_dropout(self.attention(x, x, x)) + x)
        return self.ff_norm(self.ff(attended) + x)

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, ff_size, num_blocks, num_heads, activation=nn.ReLU, dropout=None):
        """
            A single Transformer Network
            input_size: hidden weight
            hidden_size: hidden weight
            ff_size: hiden weight
        """
        super(Transformer, self).__init__()
        # construct num_blocks block, no residual structure
        self.blocks = nn.Sequential(*[Block(input_size, hidden_size, num_heads, activation, dropout=dropout)
                                      for _ in range(num_blocks)])

    def forward(self, x):
        """
            Sequentially applies the blocks of the Transformer network
        """
        return self.blocks(x)


class Net(nn.Module):
    """
        A neural network that encodes a sequence
        using a Transformer network
    """

    def __init__(self, embeddings, max_length, model_size=128, num_heads=4, num_blocks=1, dropout=0.1,
                 train_word_embeddings=True):
        super(Net, self).__init__()
        # Creates Embedding instance from given 2-dimensional FloatTensor.
        # embeddings (Tensor): FloatTensor containing weights for the Embedding.
        # First dimension is being passed to Embedding as 'num_embeddings', second as 'embedding_dim'.
        # freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
        # Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=not train_word_embeddings)
        self.model_size = model_size
        # Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
        # outputsize=[embedding.size(1), self.model_size]

        self.emb_ff = nn.Linear(embeddings.size(1), self.model_size)
        self.pos = nn.Linear(max_length, self.model_size)
        self.max_length = max_length
        self.transformer = Transformer(self.model_size, self.model_size, self.model_size, num_blocks, num_heads,
                                       dropout=dropout)
        # 2: biclass
        self.output = nn.Linear(self.model_size, 2)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1)
        x = self.emb_ff(self.embeddings(x))
        pos = self.pos(get_pos_onehot(self.max_length).to(x)).unsqueeze(0)
        x = x.view(*(x_size + (self.model_size,)))
        x += pos
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.output(x)

try:
    # try to import tqdm for progress updates
    from tqdm import tqdm
except ImportError:
    # on failure, make tqdm a noop
    def tqdm(x):
        return x

try:
    # try to import visdom for visualisation of attention weights
    import visdom
    from helpers import plot_weights

    vis = visdom.Visdom()
except ImportError:
    vis = None
    pass

def val(model, test, vocab, device, epoch_num, path_saving):
    """
        Evaluates model on the test set
    """
    # model.eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will
    # work in eval model instead of training mode.

    model.eval()
    print("\nValidating..")
    if not vis is None:
        visdom_windows = None
    # impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations but you
    # won’t be able to backprop (which you don’t want in an eval script).
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for i, b in enumerate(tqdm(test)):
            if not vis is None and i == 0:
                visdom_windows = plot_weights(model, visdom_windows, b, vocab, vis)
            model_out = model(b.review[0].to(device)).to("cpu").numpy()
            correct += (model_out.argmax(axis=1) == b.sentiment.numpy()).sum()
            total += b.sentiment.size(0)
        with open(path_saving + '_val_results', 'a', encoding='utf-8') as file:
            temp = "epoach:{}, correct:{}%, correct samples/total samples{}/{}".format(epoch_num, correct / total,
                                                                                       correct, total)
            file.write(temp + '\n')
        print(temp)
    return correct / total


def train(max_length, model_size, epochs, learning_rate, device, num_heads, num_blocks, dropout, train_word_embeddings,
          batch_size, save_path):
    """
        Trains the classifier on the IMDB sentiment dataset
    """
    # train: train iterator
    # test: test iterator
    # vectors: train data word vector
    # vocab: train data vocab
    train, test, vectors, vocab = get_imdb(batch_size, max_length=max_length)
    # creat the transformer net
    model = Net(model_size=model_size, embeddings=vectors, max_length=max_length, num_heads=num_heads,
                num_blocks=num_blocks, dropout=dropout, train_word_embeddings=train_word_embeddings).to(device)

    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_correct = 0
    with open(save_path + '_train_results', 'a', encoding='utf-8') as file_re:
        for i in range(0, epochs + 1):
            loss_sum = 0.0
            model.train()
            # train data has been spited many batch, tadm: print progress bar
            for j, b in enumerate(iter(tqdm(train))):
                optimizer.zero_grad()
                model_out = model(b.review[0].to(device))
                # calculate loss
                loss = criterion(model_out, b.sentiment.to(device))
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            print('\n **********************************************')
            loss_temp = "Epoch: {}, Loss mean: {}\n".format(i, loss_sum / j)
            file_re.write(loss_temp + '\n')
            print(loss_temp)
            # Validate on test-set every epoch
            if i % 5 == 0:
                val_correct = val(model, test, vocab, device, i, save_path)
            if val_correct > best_correct:
                best_correct = val_correct
                best_model = model
    torch.save(best_model, save_path + '_model.pkl')

if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Train a Transformer network for sentiment analysis")
    ap.add_argument("--max_length", default=500, type=int, help="Maximum sequence length, sequences longer than this \
                                                                are truncated")
    ap.add_argument("--model_size", default=128, type=int, help="Hidden size for all hidden layers of the model")
    ap.add_argument("--epochs", default=50, type=int, help="Number of epochs to train for")
    ap.add_argument("--learning_rate", default=0.0001, type=float, dest="learning_rate",
                    help="Learning rate for optimizer")
    ap.add_argument("--device", default="cuda:0", dest="device", help="Device to use for training and evaluation \
                                                                      e.g. (cpu, cuda:0)")
    ap.add_argument("--num_heads", default=4, type=int, dest="num_heads", help="Number of attention heads in the \
                                                                               Transformer network")
    ap.add_argument("--num_blocks", default=1, type=int, dest="num_blocks",
                    help="Number of blocks in the Transformer network")
    ap.add_argument("--dropout", default=0.5, type=float, dest="dropout", help="Dropout (not keep_prob, but probability \
                                                            of ZEROING during training, i.e. keep_prob = 1 - dropout)")
    ap.add_argument("--train_word_embeddings", type=bool, default=True, dest="train_word_embeddings",
                    help="Train GloVE word embeddings")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size")
    ap.add_argument("--save_path", default=r"D:\ruin\data\\" +
                                           time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                    dest="save_path",
                    help="The path to save the results")
    args = vars(ap.parse_args())

    train(**args)