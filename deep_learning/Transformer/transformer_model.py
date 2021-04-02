import os
import numpy as np
import math
import matplotlib.pyplot as plt
import sentencepiece as spm
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

""" sinusoid position encoding """
def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    ## 각 position별 hidden index별 angle 값을 구함
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin
    ## hidden 짝수 index의 angle 값의 sin값을 함
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos
    ## hidden 홀수 index의 angle 값의 cos값을 함
    return sinusoid_table

""" attention pad mask """
## Attention을 구할 때 Padding 부분을 제외하기 위한 Mask를 구하는 함수.
def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad)
    ## K의 값 중에 Pad인 부분을 True로 변경. (Sequence 부분은 False)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    ## 구해진 값의 크기를 Q-len, K-len이 되도록 변경.
    return pad_attn_mask

""" attention decoder mask """
## decoder의 현재단어와 이전단어는 볼 수 있고 다음단어는 볼 수 없도록 Masking하는 함수.
## 대충 대각선 형태로 masking이 된다고 생각
def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    ## 모든 값이 1인 Q_len, K_len 테이블 생성
    subsequent_mask = subsequent_mask.triu(diagonal=1)
    ## 대각선을 기준으로 아래쪽을 0으로 만듦.
    return subsequent_mask


""" scale dot product attention """


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout, d_head):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / (d_head ** 0.5)

    def forward(self, Q, K, V, attn_mask):
        # (bs, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2))
        ## Q * K.transpose를 구함
        scores = scores.mul_(self.scale)
        ## K-dimension에 루트를 취한 값으로 나눠줌.
        scores.masked_fill_(attn_mask, -1e9)
        ## mask 적용
        # (bs, n_head, n_q_seq, n_k_seq)
        attn_prob = nn.Softmax(dim=-1)(scores)
        ## softmax를 취해 각 단어의 가중치 확률분포 attn_prob를 구함
        attn_prob = self.dropout(attn_prob)
        # (bs, n_head, n_q_seq, d_v)
        context = torch.matmul(attn_prob, V)
        ## attn_prob * V를 구함. 구한 값은 Q에 대한 V의 가중치 합 벡터.
        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn_prob


""" multi head attention """


class MultiHeadAttention(nn.Module):
    def __init__(self, d_hidn, n_head, d_head, dropout):
        super().__init__()
        self.d_hidn = d_hidn
        self.n_head = n_head
        self.d_head = d_head
        self.W_Q = nn.Linear(d_hidn, n_head * d_head)
        self.W_K = nn.Linear(d_hidn, n_head * d_head)
        self.W_V = nn.Linear(d_hidn, n_head * d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(dropout, d_head)
        self.linear = nn.Linear(n_head * d_head, d_hidn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        # (bs, n_head, n_q_seq, d_head)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        ## Q * W_Q를 한 후 multi-head로 나눔.
        # (bs, n_head, n_k_seq, d_head)
        k_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        ## K * W_K를 한 후 multi-head로 나눔.
        # (bs, n_head, n_v_seq, d_head)
        v_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        ## V * W_V를 한 후 multi-head로 나눔.
        # (bs, n_head, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        ## 어텐션 마스크도 멀티 헤드로 변경.
        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        ## 스케일닷 프로덕트 어텐션 클래스를 이용해 각 head별 어텐션을 구함.
        # (bs, n_head, n_q_seq, h_head * d_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_head)
        ## 여러개의 head를 하나로 합침.
        # (bs, n_head, n_q_seq, e_embd)
        output = self.linear(context)
        ## linear를 취해 최종 멀티 헤드 어텐션 값을 구함.
        output = self.dropout(output)
        # (bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn_prob

## 포지션 와이스는 간단. 대충 줄였다 늘렸다 정도로 생각.
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_hidn, d_ff, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(d_hidn, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_hidn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        x = self.dropout(torch.relu(self.fc_1(x)))
        # x: [batch_size, seq_len, pf_dim]
        x = self.fc_2(x)
        x = self.dropout(x)
        # x: [batch_size, seq_len, hidden_dim]
        return x


## 인코더에서 루프를 돌며 처리할 수 있도록 인코더 레이어를 정의하고 여러개 만들어 실행.
class EncoderLayer(nn.Module):
    def __init__(self, d_hidn, n_head, d_head, dropout, d_ff, layer_norm_epsilon):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_hidn, n_head, d_head, dropout)
        self.layer_norm1 = nn.LayerNorm(d_hidn, eps=layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(d_hidn, d_ff, dropout)
        self.layer_norm2 = nn.LayerNorm(d_hidn, eps=layer_norm_epsilon)
        ## 여기는 인코더 순서대로 정의.
        ## 멀티헤드 어텐션 -> add & norm -> Feed Forward -> add & norm
    def forward(self, inputs, attn_mask):
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        ## 멀티 헤드 어텐션 수행
        att_outputs = self.layer_norm1(inputs + att_outputs)
        ## 위의 결과 att_outputs와 원래의 inputs을 더한 후 LayerNorm을 실행
        # (bs, n_enc_seq, d_hidn)
        ffn_outputs = self.pos_ffn(att_outputs)
        ## 위의 결과를 입력으로 Feed Forward를 실행
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        ## 위의 결과와 att_outputs를 더한후 LayerNorm을 실행
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        return ffn_outputs, attn_prob


""" encoder """
## 인코더 클래스. 위의 인코더 레이어를 사용해서 여러번 반복한다고 생각.
class Encoder(nn.Module):
    def __init__(self, n_enc_vocab, d_hidn, n_enc_seq, n_head, d_head, dropout,
                 d_ff, layer_norm_epsilon, n_layer, i_pad):
        super().__init__()
        self.i_pad = i_pad
        self.enc_emb = nn.Embedding(n_enc_vocab, d_hidn)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(n_enc_seq + 1, d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(d_hidn, n_head, d_head, dropout, d_ff, layer_norm_epsilon) for _ in range(n_layer)])
        ## 레이어 개수만큼의 인코더층을 사용하므로 여러번 쌓는 코드를 구현.
        ## nn.ModuleList는 nn.Module을 리스트로 정리하는 방법이다.
        ## 각 레이어를 리스트에 전달하고 레이어의 이터레이터를 만든다.
        ## 설명 : https://michigusa-nlp.tistory.com/26
    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(self.i_pad)
        positions.masked_fill_(pos_mask, 0)
        ## 입력에 대한 position 값을 구함. 포지셔널 인코딩과 input 임베딩을 더하기 위한 전과정
        # (bs, n_enc_seq, d_hidn)
        outputs = self.enc_emb(inputs) + self.pos_emb(positions)
        ## input embedding과 position embedding을 구한 후 더함.
        # (bs, n_enc_seq, n_enc_seq)
        attn_mask = get_attn_pad_mask(inputs, inputs, self.i_pad)
        ## 입력에 대한 attention pad mask를 구함.
        ## attention pad mask는 어텐션을 구할 때 padding된 부분을 제외하기 위해 쓰는 것.
        attn_probs = []
        for layer in self.layers:
            ## 트랜스포머는 num_layers 개수만큼의 인코더층을 사용하므로 이를 여러번 쌓는 코드를 별도 구현하는 것.
            # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
            outputs, attn_prob = layer(outputs, attn_mask)
            ## for 루프를 돌며 각 layer를 실행. layer의 입력은 이전 layer의 출력값.
            attn_probs.append(attn_prob)
        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return outputs, attn_probs


""" decoder layer """
## 인코더 레이어와 마찬가지로 디코더 레이어도 순서대로 정의.
class DecoderLayer(nn.Module):
    def __init__(self, d_hidn, n_head, d_head, dropout, d_ff, layer_norm_epsilon):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_hidn, n_head, d_head, dropout)
        self.layer_norm1 = nn.LayerNorm(d_hidn, eps=layer_norm_epsilon)
        self.dec_enc_attn = MultiHeadAttention(d_hidn, n_head, d_head, dropout)
        self.layer_norm2 = nn.LayerNorm(d_hidn, eps=layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(d_hidn, d_ff, dropout)
        self.layer_norm3 = nn.LayerNorm(d_hidn, eps=layer_norm_epsilon)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, dec_enc_attn_mask):
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_dec_seq)
        self_att_outputs, self_attn_prob = self.self_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        ## 멀티 헤드 어텐션 수행. Q, K, V 모두 동일한 값을 사용하는 셀프 어텐션.
        self_att_outputs = self.layer_norm1(dec_inputs + self_att_outputs)
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_enc_seq)
        dec_enc_att_outputs, dec_enc_attn_prob = self.dec_enc_attn(self_att_outputs, enc_outputs, enc_outputs,
                                                                   dec_enc_attn_mask)
        ## 인코더-디코더 멀티 헤드 어텐션 수행. 각각 Query, Key, Value로 들어가는데
        ## Query는 위 디코더에서 구한 결과고 Key, Value는 인코더에서 구한 결과가 들어간다.
        dec_enc_att_outputs = self.layer_norm2(self_att_outputs + dec_enc_att_outputs)
        # (bs, n_dec_seq, d_hidn)
        ffn_outputs = self.pos_ffn(dec_enc_att_outputs)
        ffn_outputs = self.layer_norm3(dec_enc_att_outputs + ffn_outputs)
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_dec_seq), (bs, n_head, n_dec_seq, n_enc_seq)
        return ffn_outputs, self_attn_prob, dec_enc_attn_prob


""" decoder """


class Decoder(nn.Module):
    def __init__(self, n_dec_vocab, d_hidn, n_dec_seq, n_layer, n_head, d_head,
                 dropout, d_ff, layer_norm_epsilon, i_pad):
        super().__init__()
        self.i_pad = i_pad
        self.dec_emb = nn.Embedding(n_dec_vocab, d_hidn)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(n_dec_seq + 1, d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([DecoderLayer(d_hidn, n_head, d_head, dropout, d_ff, layer_norm_epsilon) for _ in range(n_layer)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device, dtype=dec_inputs.dtype).expand(
            dec_inputs.size(0), dec_inputs.size(1)).contiguous() + 1
        pos_mask = dec_inputs.eq(self.i_pad)
        positions.masked_fill_(pos_mask, 0)
        ## 인코더에서와 마찬가지로 입력에 대한 position 값을 구함.
        # (bs, n_dec_seq, d_hidn)
        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(positions)
        ## input embedding과 position embedding을 구한 후 더함.
        # (bs, n_dec_seq, n_dec_seq)
        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.i_pad)
        ## 입력에 대한 어텐션 패드 마스크를 구함.
        ## 이 과정은 디코더 마스킹이 아니라 어텐션에 대해 패딩값이 영향을 주지 않게 하려고 하는거라 디코더에서도 함.
        # (bs, n_dec_seq, n_dec_seq)
        dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs)
        ## 입력에 대한 decoder attention mask를 구함.
        ## 인코더와 차별화되는 곳으로 현재 보는 곳의 이후 내용은 보지 못하도록 하기 위함.
        # (bs, n_dec_seq, n_dec_seq)
        ## dec_attn_pad_mask는 True, False 값 리턴
        ## dec_attn_decoder_mask는 0 혹은 1 리턴.
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)
        ## torch.gt는 input이 other보다 크면 True이고, 그 외는 False를 주는 부울 텐서
        ## dec_attn_pad_mask + dec_attn_decoder_mask 값이 0보다 크면 True를 리턴하고 같거나 작으면 False.
        ## Sequence 부분은 False, Padding 부분은 True가 들어간다. 둘을 합쳐주는 것으로 최종적인 decoder_masking이 구해짐
        # (bs, n_dec_seq, n_enc_seq)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.i_pad)
        ## Q(decoder input), K(encoder output)에 대한 어텐션 마스크를 구함.
        self_attn_probs, dec_enc_attn_probs = [], []
        for layer in self.layers:
            # (bs, n_dec_seq, d_hidn), (bs, n_dec_seq, n_dec_seq), (bs, n_dec_seq, n_enc_seq)
            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                                   dec_enc_attn_mask)
            self_attn_probs.append(self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)
            ## 인코더에서 했던 것과 마찬가지로 각 layer를 실행.
            ## layer의 입력은 이전 layer의 출력값.
        # (bs, n_dec_seq, d_hidn), [(bs, n_dec_seq, n_dec_seq)], [(bs, n_dec_seq, n_enc_seq)]S
        return dec_outputs, self_attn_probs, dec_enc_attn_probs


""" transformer """


class Transformer(nn.Module):
    def __init__(self, n_enc_vocab, n_dec_vocab, d_hidn, n_enc_seq, n_dec_seq,
                 n_head, d_head, dropout, d_ff, layer_norm_epsilon, n_layer, i_pad):
        super().__init__()
        self.encoder = Encoder(n_enc_vocab, d_hidn, n_enc_seq, n_head, d_head, dropout, d_ff, layer_norm_epsilon, n_layer, i_pad)
        self.decoder = Decoder(n_dec_vocab, d_hidn, n_dec_seq, n_layer, n_head, d_head, dropout, d_ff, layer_norm_epsilon, i_pad)

    def forward(self, enc_inputs, dec_inputs):
        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        enc_outputs, enc_self_attn_probs = self.encoder(enc_inputs)
        ## 인코더 input으로 Encoder 실행
        # (bs, n_seq, d_hidn), [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        ## 인코더 아웃풋과 디코더 인풋을 입력으로 디코더 실행.
        # (bs, n_dec_seq, n_dec_vocab), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs

from torchtext import data
from torchtext.data import TabularDataset
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 3

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

train_data, test_data = TabularDataset.splits(
        path=r"D:\ruin\data\csv_file\imdb_split", train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('테스트 샘플의 개수 : {}'.format(len(test_data)))

TEXT.build_vocab(train_data, min_freq=5) # 단어 집합 생성
LABEL.build_vocab(train_data)

n_classes = 2

vocab_size = len(TEXT.vocab)

train_data, val_data = train_data.split(split_ratio=0.8)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), sort=False,batch_size=BATCH_SIZE,
        shuffle=True, repeat=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
