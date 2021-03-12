from torchtext import data

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True,
                  fix_length=20)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

# sequential : 시퀀스 데이터 여부. (True가 기본값)
# use_vocab : 단어 집합을 만들 것인지 여부. (True가 기본값)
# tokenize : 어떤 토큰화 함수를 사용할 것인지 지정. (string.split이 기본값)
# lower : 영어 데이터를 전부 소문자화한다. (False가 기본값)
# batch_first : 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올 것인지 여부. (False가 기본값)
# is_target : 레이블 데이터 여부. (False가 기본값)
# fix_length : 최대 허용 길이. 이 길이에 맞춰서 패딩 작업(Padding)이 진행된다.

from torchtext.data import TabularDataset

train_data, test_data = TabularDataset.splits(path='.', train='train_data.csv', test='test_data.csv', format='csv',
                                              fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('테스트 샘플의 개수 : {}'.format(len(test_data)))

TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
# min_freq : 단어 집합에 추가 시 단어의 최소 등장 빈도 조건을 추가.
# max_size : 단어 집합의 최대 크기를 지정.

print('단어 집합의 크기 : {}'.format(len(TEXT.vocab)))

from torchtext.data import Iterator

batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size = batch_size)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)

batch = next(iter(train_loader))
print(type(batch))