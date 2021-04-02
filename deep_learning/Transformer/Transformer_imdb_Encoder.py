import torch
from torch import nn
import torch.nn.functional as f
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_Softargmax = nn.Softmax  # fix wrong name


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, p, d_input=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq, d_xk, d_xv = d_input

        # Make sure that the embedding dimension of model is a multiple of number of heads
        assert d_model % self.num_heads == 0

        self.d_k = d_model // self.num_heads

        # These are still of dimension d_model. They will be split into number of heads
        # - This matrix allows us to rotate the current input (see section: #Queries,-keys-and-values)
        self.W_q = nn.Linear(d_xq, d_model, bias=False)  # q = W_q*x
        self.W_k = nn.Linear(d_xk, d_model, bias=False)  # k = W_k*x
        self.W_v = nn.Linear(d_xv, d_model, bias=False)  # v = W_v*x

        # Outputs of all sub-layers need to be of dimension d_model
        # -  (see section (in cross-attetion): #Implementation)
        self.W_h = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        """
        Attention vectorization calculation: A = softargmax(K^T*Q) and
        Hidden vectorization representation: H = V*A
        (see section: #Cross-attention)
        """

        batch_size = Q.size(0)
        k_length = K.size(-2)

        # Scaling by d_k so that the soft(arg)max doesnt saturate
        Q = Q / np.sqrt(self.d_k)  # (bs, n_heads, q_length, dim_per_head)

        # -- Compute the K - Q aligment: K^T*Q
        scores = torch.matmul(Q, K.transpose(2, 3))  # (bs, n_heads, q_length, k_length)

        # -- Compute the attention: softargmax(K-Q aligment)
        A = nn_Softargmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)

        # -- Compute the hidden representation: H = V*A
        # Get the weighted average of the values
        H = torch.matmul(A, V)  # (bs, n_heads, q_length, dim_per_head)

        return H, A

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (heads X depth)
        Return after transpose to put in shape (batch_size X num_heads X seq_length X d_k)
        """
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        """
        Combine the heads again to get (batch_size X seq_length X (num_heads times d_k))
        """
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

    def forward(self, X_q, X_k, X_v):
        batch_size, seq_length, dim = X_q.size()

        # -- Gets query Q, keys K and values V from the inputs using W_q, W_k and W_v
        # After transforming, split into num_heads
        Q = self.split_heads(self.W_q(X_q), batch_size)  # (bs, n_heads, q_length, dim_per_head) -> Q = W_q*X_q
        K = self.split_heads(self.W_k(X_k), batch_size)  # (bs, n_heads, k_length, dim_per_head) -> K = W_k*X_k
        V = self.split_heads(self.W_v(X_v), batch_size)  # (bs, n_heads, v_length, dim_per_head) -> V = W_v*X_v

        # -- Calculate the attention weights and hidden representations for each of the heads
        H_cat, A = self.scaled_dot_product_attention(Q, K, V)

        # Put all the heads back together by concat
        H_cat = self.group_heads(H_cat, batch_size)  # (bs, q_length, dim)

        # -- Collapse the h-long hidden representation vector in the model dimension using W_h
        # -  (see section (in cross-attetion): #Implementation)
        # Final linear layer
        H = self.W_h(H_cat)  # (bs, q_length, dim)

        return H, A

class CNN(nn.Module):
    def __init__(self, d_model, hidden_dim, p):
        super().__init__()
        self.k1convL1 = nn.Linear(d_model,    hidden_dim)
        self.k1convL2 = nn.Linear(hidden_dim, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.k1convL1(x)
        x = self.activation(x)
        x = self.k1convL2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, conv_hidden_dim, p=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, p)  # Self-attetion = Cross-attetion with q=k=v=x
        self.cnn = CNN(d_model, conv_hidden_dim, p)

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

    def forward(self, x):
        # Self attention
        attn_output, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)

        # Layer norm after adding the residual connection
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        # Feed forward
        cnn_output = self.cnn(out1)  # (batch_size, input_seq_len, d_model)

        # Second layer norm after adding residual connection
        out2 = self.layernorm2(out1 + cnn_output)  # (batch_size, input_seq_len, d_model)

        return out2


def create_sinusoidal_embeddings(nb_p, dim, E):
    theta = np.array([
        [p / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for p in range(nb_p)
    ])
    E[:, 0::2] = torch.FloatTensor(np.sin(theta[:, 0::2]))
    E[:, 1::2] = torch.FloatTensor(np.cos(theta[:, 1::2]))
    E.detach_()
    E.requires_grad = False
    E = E.to(device)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size, max_position_embeddings, p):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        create_sinusoidal_embeddings(
            nb_p=max_position_embeddings,
            dim=d_model,
            E=self.position_embeddings.weight
        )

        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        # Get word embeddings for each input id
        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)

        # Get position embeddings for each position id
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        # Add them both
        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)

        # Layer norm
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ff_hidden_dim, input_vocab_size,
                 maximum_position_encoding, p=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embeddings(d_model, input_vocab_size, maximum_position_encoding, p)

        # We can stick different encoder, one after other, to have a Deep Neural network.
        self.enc_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.enc_layers.append(EncoderLayer(d_model, num_heads, ff_hidden_dim, p))

    def forward(self, x):
        x = self.embedding(x)  # Transform to (batch_size, input_seq_length, d_model)

        # We feed forward one transformer encoder after other
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # (batch_size, input_seq_len, d_model)

import torchtext.data as data
import torchtext.datasets as datasets
from torchtext.data import TabularDataset

max_len = 200
text = data.Field(sequential=True, fix_length=max_len, batch_first=True, lower=True, dtype=torch.long)
label = data.LabelField(sequential=False, dtype=torch.long)
datasets.IMDB.download('./')
ds_train, ds_test = datasets.IMDB.splits(text, label, path='./imdb/aclImdb/')
print('train : ', len(ds_train))
print('test : ', len(ds_test))
print('train.fields :', ds_train.fields)

ds_train, ds_valid = ds_train.split(0.9)
print('train : ', len(ds_train))
print('valid : ', len(ds_valid))
print('test : ', len(ds_test))

num_words = 50_000
text.build_vocab(ds_train, max_size=num_words)
label.build_vocab(ds_train)
vocab = text.vocab

batch_size = 128
train_loader, valid_loader, test_loader = data.BucketIterator.splits(
    (ds_train, ds_valid, ds_test), batch_size=batch_size, sort_key=lambda x: len(x.text), repeat=False)


class TransformerClassifier(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, conv_hidden_dim, input_vocab_size, num_answers):
        super().__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, conv_hidden_dim, input_vocab_size,
                               maximum_position_encoding=10000)
        self.dense = nn.Linear(d_model, num_answers)

    def forward(self, x):
        x = self.encoder(x)

        x, _ = torch.max(x, dim=1)
        x = self.dense(x)
        return x

model = TransformerClassifier(num_layers=1, d_model=32, num_heads=2,
                         conv_hidden_dim=128, input_vocab_size=50002, num_answers=2)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
epochs = 15
t_total = len(train_loader) * epochs


def train(train_loader, valid_loader):
    for epoch in range(epochs):
        train_iterator, valid_iterator = iter(train_loader), iter(valid_loader)
        nb_batches_train = len(train_loader)
        train_acc = 0
        model.train()
        losses = 0.0

        for batch in train_iterator:
            x = batch.text.to(device)
            y = batch.label.to(device)

            # Perform the forward pass of the model
            out = model(x)  # ①

            # Get loss
            loss = f.cross_entropy(out, y)  # ②

            # Clear gradient buffers
            model.zero_grad()  # ③

            # Calculate gradient
            loss.backward()  # ④
            losses += loss.item()

            # Perform training setp
            optimizer.step()  # ⑤

            train_acc += (out.argmax(1) == y).cpu().numpy().mean()

        print(f"Training loss at epoch {epoch} is {losses / nb_batches_train}")
        print(f"Training accuracy: {train_acc / nb_batches_train}")
        print('Evaluating on validation:')
        evaluate(valid_loader)


def evaluate(data_loader):
    data_iterator = iter(data_loader)
    nb_batches = len(data_loader)
    model.eval()
    acc = 0
    for batch in data_iterator:
        x = batch.text.to(device)
        y = batch.label.to(device)

        out = model(x)
        acc += (out.argmax(1) == y).cpu().numpy().mean()

    print(f"Test accuracy: {acc / nb_batches}")

train(train_loader, valid_loader)
evaluate(test_loader)


## https://github.com/victor-roris/pytorch-Deep-Learning/blob/9b3bf8a3381186f80dc406e756b0262a010320a2/15-transformer.ipynb