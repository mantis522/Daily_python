import torch
import torch.nn as nn
import torch.optim as optim

sentence = "Repeat is the best medicine for memory".split()

vocab = list(set(sentence))

word2index = {tkn: i for i, tkn in enumerate(vocab, 1)}  # 단어에 고유한 정수 부여
word2index['<unk>'] = 0


index2word = {v: k for k, v in word2index.items()}

def build_data(sentence, word2index):
    encoded = [word2index[token] for token in sentence] # 각 문자를 정수로 변환.
    input_seq, label_seq = encoded[:-1], encoded[1:] # 입력 시퀀스와 레이블 시퀀스를 분리
    input_seq = torch.LongTensor(input_seq).unsqueeze(0) # 배치 차원 추가
    label_seq = torch.LongTensor(label_seq).unsqueeze(0) # 배치 차원 추가
    return input_seq, label_seq

X, Y = build_data(sentence, word2index)

class Net(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, batch_first=True):
        super(Net, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        self.rnn_layer = nn.RNN(input_size, hidden_size, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, vocab_size) ## 입력은 hidden_size의 크기, 출력은 원-핫 벡터의 크기.

    def forward(self,x):
        output = self.embedding_layer(x)
        output, hidden = self.rnn_layer(output)
        output = self.linear(output)

        return output.view(-1, output.size(2)) ## torch의 view 함수에서 -1은 토치가 알아서 해달라는 의미. 여기에서는 output이라는 텐서를 (?, output.size(2))로 변경하라는 의미.

# 하이퍼 파라미터
vocab_size = len(word2index)  # 단어장의 크기는 임베딩 층, 최종 출력층에 사용된다. <unk> 토큰을 크기에 포함한다.
input_size = 5  # 임베딩 된 차원의 크기 및 RNN 층 입력 차원의 크기
hidden_size = 20  # RNN의 은닉층 크기

model = Net(vocab_size, input_size, hidden_size, batch_first=True)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters()) # 모델의 학습 가능한 매개변수(예. 가중치와 편향)들은 모델의 매개변수에 포함되어 있음

# 임의로 예측해보기. 가중치는 전부 랜덤 초기화 된 상태이다.
output = model(X)
print(output)

print(output.shape) ## torch.Size([6, 8])
## 예측값의 크기는 (6, 8)입니다. 이는 각각 (시퀀스의 길이, 은닉층의 크기)에 해당됩니다. 모델은 훈련시키기 전에 예측을 제대로 하고 있는지 예측된 정수 시퀀스를 다시 단어 시퀀스로 바꾸는 decode 함수를 만듭니다.

# 수치화된 데이터를 단어로 전환하는 함수
decode = lambda y: [index2word.get(x) for x in y]
print(decode)

# 훈련 시작
for step in range(201):
    # 경사 초기화
    optimizer.zero_grad()
    # 순방향 전파
    output = model(X)
    # 손실값 계산
    loss = loss_function(output, Y.view(-1))
    # 역방향 전파
    loss.backward()
    # 매개변수 업데이트
    optimizer.step()
    # 기록
    if step % 40 == 0:
        print("[{:02d}/201] {:.4f} ".format(step+1, loss))
        pred = output.softmax(-1).argmax(-1).tolist()
        print(" ".join(["Repeat"] + decode(pred)))
        print()