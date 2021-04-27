import random



def making_label():
    test_true = []
    test_false = []
    for a in range(50):
        b = random.random()
        if b < 0.5:
            test_true.append(b)
        else:
            test_false.append(b)

    return test_true, test_false

test_true, test_false = making_label()

