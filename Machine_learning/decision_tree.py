# 판다스 버전
import pandas as pd
import numpy as np

# data = pd.DataFrame({
#                      "edu":['d', 'd', 'u', 'm', 'm', 'd', 'd', 'm', 'u', 'u', 'd', 'u'],
#                      "age":['60', '40', '39', '39', '40', '60', '39', '39', '40', '40', '39', '39'],
#                      "health":['m', 'l', 'h', 'l', 'm', 'l', 'h', 'l', 'm', 'l', 'h', 'h'],
#                      'saving':['y', 'n', 'n', 'y', 'n', 'y', 'n', 'n', 'y', 'y', 'y', 'n'],
#                      'class':['y', 'n', 'n', 'y', 'n', 'y', 'n', 'n', 'y', 'y', 'n', 'n']
#
#                      },
#
#
#                     columns=['edu', 'age', 'health', 'saving', 'class'])

data = pd.DataFrame({
                     "edu":['u', 'm', 'u', 'd'],
                     "age":['39', '39', '40', '60'],
                     "health":['h', 'l', 'm', 'h'],
                     'saving':['n', 'y', 'y', 'y'],
                     'class':['y', 'n', 'y', 'y']

                     },


                    columns=['edu', 'age', 'health', 'saving', 'class'])


# 기술 속성(descriptive features)
features = data[['edu', 'age', 'health', 'saving']]
# 대상 속성(target feature)
target = data['class']
print(data)

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts = True)
    entropy = -np.sum([(counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


def InfoGain(data, split_attribute_name, target_name):
    # 전체 엔트로피 계산
    total_entropy = entropy(data[target_name])
    print('Entropy(D) = ', round(total_entropy, 5))

    # 가중 엔트로피 계산
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum([(counts[i] / np.sum(counts)) *
                               entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
                               for i in range(len(vals))])
    print('H(', split_attribute_name, ') = ', round(Weighted_Entropy, 5))

    # 정보이득 계산
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


# print('InfoGain( Height ) = ', round(InfoGain(data, "Height", "class"), 5), '\n')
# print('InfoGain( Income ) = ', round(InfoGain(data, "Income", "class"), 5), '\n')
# print('InfoGain( Education ) = ', round(InfoGain(data, "Education", "class"), 5), '\n')


def ID3(data, originaldata, features, target_attribute_name, parent_node_class=None):
    # 중지기준 정의

    # 1. 대상 속성이 단일값을 가지면: 해당 대상 속성 반환
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # 2. 데이터가 없을 때: 원본 데이터에서 최대값을 가지는 대상 속성 반환
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name]) \
            [np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    # 3. 기술 속성이 없을 때: 부모 노드의 대상 속성 반환
    elif len(features) == 0:
        return parent_node_class

    # 트리 성장
    else:
        # 부모노드의 대상 속성 정의(예: Good)
        parent_node_class = np.unique(data[target_attribute_name]) \
            [np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        # 데이터를 분할할 속성 선택
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # 트리 구조 생성
        tree = {best_feature: {}}

        # 최대 정보이득을 보인 기술 속성 제외
        features = [i for i in features if i != best_feature]

        # 가지 성장
        for value in np.unique(data[best_feature]):
            # 데이터 분할. dropna(): 결측값을 가진 행, 열 제거
            sub_data = data.where(data[best_feature] == value).dropna()

            # ID3 알고리즘
            subtree = ID3(sub_data, data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

        return (tree)

tree = ID3(data, data, ['edu', 'age', 'health', 'saving'], 'class')
from pprint import pprint
pprint(tree)
