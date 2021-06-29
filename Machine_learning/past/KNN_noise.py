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

def select_train_data(x_data):
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

def making_noise():
    true_noise_data = []
    true_noise_y_data = []
    false_noise_data = []
    false_noise_y_data = []
    for a in range(30):
        random_number = rangeNum()
        if random_number < 0.5:
            false_noise_data.append(random_number)
            false_noise_y_data.append(rangeNum())
        else:
            true_noise_data.append(random_number)
            true_noise_y_data.append(rangeNum())

    return true_noise_data[:1], true_noise_y_data[:1], false_noise_data[:1], false_noise_y_data[:1]

def making_label(data):
    x_label = []
    for a in range(len(data)):
        if data[a] < 0.5:
            x_label.append(1)
        else:
            x_label.append(0)

    return x_label

def making_noise_label(data):
    x_label = []
    for a in range(len(data)):
        if data[a] < 0.5:
            x_label.append(0)
        else:
            x_label.append(1)

    return x_label

def merge_list(*args, fill_value = None):
    max_length = max([len(lst) for lst in args])
    merged = []
    for i in range(max_length):
        merged.append([
        args[k][i] if i < len(args[k]) else fill_value for k in range(len(args))
        ])
    return merged

def slice_data(number, true_train_data, false_train_data, true_label, false_label):
    train_data = np.concatenate((true_train_data[:number], false_train_data[:number]), axis=0)
    label = true_label[:number] + false_label[:number]
    return train_data, label

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

def accuracy(train_data, train_label, test_data, test_label, n):
    count = 0
    for a in range(100):
        result = classify0(test_data[a], train_data, train_label, n)
        if result == test_label[a]:
            count = count + 1

    return count / 100

def cal_accuracy():
    sum = []
    for a in range(100):
        x_data, y_data = making_data(30)
        test_data, test_y_data = making_data(100)
        true_data, true_y_data, false_data, false_y_data = select_train_data(x_data)
        true_noise_data, true_noise_y_data, false_noise_data, false_noise_y_data = making_noise()
        true_label = making_label(true_data)
        false_label = making_label(false_data)
        true_noise_label = making_noise_label(true_noise_data)
        false_noise_label = making_noise_label(false_noise_data)
        test_label = making_label(test_data)
        true_train_data = np.array(merge_list(true_data, true_y_data))
        false_train_data = np.array(merge_list(false_data, false_y_data))
        test_data = np.array(merge_list(test_data, test_y_data))
        true_noise_data = np.array(merge_list(true_noise_data, true_noise_y_data))
        false_noise_data = np.array(merge_list(false_noise_data, false_noise_y_data))
        train_data, train_label = slice_data(9, true_train_data, false_train_data, true_label, false_label)
        concat_train_data = np.concatenate((train_data, true_noise_data), axis=0)
        concat_train_data = np.concatenate((concat_train_data, false_noise_data), axis=0)
        concat_train_label = train_label + true_noise_label
        concat_train_label = concat_train_label + false_noise_label
        knn_accuracy = accuracy(concat_train_data, concat_train_label, test_data, test_label, 3)
        sum.append(knn_accuracy)

    mean = np.mean(sum)
    std = np.std(sum)

    return round(mean, 5), round(std, 5)

mean, std  = cal_accuracy()

print("mean : ", mean)
print("std : ", std)

