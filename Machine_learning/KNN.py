import numpy as np
import operator
import random

def rangeNum():
    return float(random.random())

def making_data(number):
    x_list = []
    y_list = []
    for a in range(number):
        x_list.append(rangeNum())
        y_list.append(rangeNum())

    return x_list, y_list

x_data, y_data = making_data(30)
test_data, test_y_data = making_data(100)

def select_train_data():
    true_data = []
    true_y_data = []
    false_data = []
    false_y_data = []
    for a in range(len(x_data)):
        if x_data[a] < 0.5:
            true_data.append(x_data[a])
            true_y_data.append(rangeNum())
        else:
            false_data.append(x_data[a])
            false_y_data.append(rangeNum())

    return true_data, true_y_data, false_data, false_y_data

true_data, true_y_data, false_data, false_y_data = select_train_data()

def making_label(data):
    x_label = []
    for a in range(len(data)):
        if data[a] < 0.5:
            x_label.append(1)
        else:
            x_label.append(0)

    return x_label

true_label = making_label(true_data)
false_label = making_label(false_data)

test_label = making_label(test_data)

def merge_list(*args, fill_value = None):
    max_length = max([len(lst) for lst in args])
    merged = []
    for i in range(max_length):
        merged.append([
        args[k][i] if i < len(args[k]) else fill_value for k in range(len(args))
        ])
    return merged

true_train_data = np.array(merge_list(true_data, true_y_data))
false_train_data = np.array(merge_list(false_data, false_y_data))
test_data = np.array(merge_list(test_data, test_y_data))

def slice_data(number):
    train_data = np.concatenate((true_train_data[:number], false_train_data[:number]), axis=0)
    label = true_label[:number] + false_label[:number]
    return train_data, label

train_data, train_label = slice_data(2)
print(train_data)
print(train_label)

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

def KNN(epochs):
    for a in range(epochs):
        result = classify0(test_data[a], train_data, train_label, 1)
        print(a, "예측값 : ", result)
        print(a, "실제값 : ", test_label[a])

a = KNN(100)

print(a)

result = classify0([0.5, 0.1], train_data, train_label, 1)
# print(result)