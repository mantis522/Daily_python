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

x_label = making_label()

print(x_label)

def merge_list(*args, fill_value = None):
    max_length = max([len(lst) for lst in args])
    merged = []
    for i in range(max_length):
        merged.append([
        args[k][i] if i < len(args[k]) else fill_value for k in range(len(args))
        ])
    return merged

train_data = merge_list(x_data, y_data)
