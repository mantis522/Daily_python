import numpy as np
import operator
import random

def rangeNum(s, e):
    return float(random.random() * (e - s + 1)) + s

def making_data(number):
    x_list = []
    y_list = []
    for a in range(number):
        x_list.append(rangeNum(0, 0))
        y_list.append(rangeNum(0, 0))

    return x_list, y_list

x_data, y_data = making_data(10000)

print(x_data)
print(y_data)

def making_label():
    x_label = []
    for a in range(len(x_data)):
        if x_data[a] < 0.5:
            x_label.append(1)
        else:
            x_label.append(0)

    return x_label

def merge_list(*args, fill_value = None):
    max_length = max([len(lst) for lst in args])
    merged = []
    for i in range(max_length):
        merged.append([
        args[k][i] if i < len(args[k]) else fill_value for k in range(len(args))
        ])
    return merged

train_data = np.array(merge_list(x_data, y_data))
train_label = making_label()

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

result = classify0([0.5, 0.1], train_data, train_label, 1)
print(result)