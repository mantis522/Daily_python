import pandas as pd

df = pd.read_csv(r"C:\Users\ruin\Desktop\class\Machine Learning\dataset\iris.data", header=None)

df.replace('Iris-setosa', 0, inplace=True)
df.replace('Iris-versicolor', 1, inplace=True)
df.replace('Iris-virginica', 2, inplace=True)

from sklearn import tree

feature = df.loc[:, [0, 1, 2, 3]]
target = df.loc[:, [4]]


from sklearn import datasets
iris = datasets.load_iris()

import pandas as pd
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['target'] = iris.target

import seaborn as sns
sns.pairplot(df, hue='target')

from sklearn.model_selection import train_test_split

feature = df.loc[:, ['sepal length (cm)', 'petal length (cm)']]
target = df.loc[:, ['target']]

x_feature, y_feature, x_target, y_target = train_test_split(feature, target, train_size=0.9, random_state=2)
x_target.head()

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(x_feature)
x_feature_std = sc.transform(x_feature)
y_feature_std = sc.transform(y_feature)
print(x_feature_std)

from sklearn import svm

clf_s = svm.SVC(kernel='linear', C=1)
clf_s.fit(x_feature_std, x_target)
predicted = clf_s.predict(y_feature_std)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_target, predicted)

import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

plt.style.use('ggplot')
feature_combined_std = np.vstack((x_feature_std, y_feature_std))
target_combined = np.hstack((x_target.values.T, y_target.values.T))
target_combined = np.reshape(target.combine, (-1))

# fig = plt.figure(figsize(13, 8)) ## IPython에서만?
# plot_decision_regions(feature_combined_std, target_combined, clf=clf_s, res=0.02)
# plt.show()

from sklearn.naive_bayes import GaussianNB
clf_n = GaussianNB()
clf_n = clf_n.fit(x_feature_std, x_target)
predicted_n = clf_n.predict(y_feature_std)

confusion_matrix(y_target, predicted_n)

from sklearn.linear_model import PassiveAggressiveClassifier

clf_p = PassiveAggressiveClassifier()
clf_p = clf_p.fit(x_feature_std, x_target)
predicted_p = clf_p.predict(y_feature_std)

confusion_matrix(y_target, predicted_p)

df_s = pd.read_csv(r"C:\Users\ruin\Desktop\class\Machine Learning\dataset\sample.csv", encoding='utf-8')
df_s.head()

from sklearn.feature_extraction.text import CountVectorizer

x_train, y_train, x_target, y_target = train_test_split(df_s['Column2'], df_s['Column1'], train_size=0.8)
vectorizer = CountVectorizer()
vectorizer.fit(x_train)

x_train_bow = vectorizer.transform(x_train)
y_train_bow = vectorizer.transform(y_train)

from sklearn.naive_bayes import BernoulliNB

clf_b = BernoulliNB()
clf_b = clf_b.fit(x_train_bow, x_target)
predicted_b = clf_b.predict(y_train_bow)

b = confusion_matrix(y_target, predicted_b)

print(b)