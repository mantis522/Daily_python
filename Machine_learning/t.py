import pandas as pd

df = pd.read_csv(r"C:\Users\ruin\Desktop\class\Machine Learning\dataset\iris.data", header=None)

df.replace('Iris-setosa', 0, inplace=True)
df.replace('Iris-versicolor', 1, inplace=True)
df.replace('Iris-virginica', 2, inplace=True)

from sklearn import tree

feature = df.loc[:, [0, 1, 2, 3]]
target = df.loc[:, [4]]
a = feature.head(5)


