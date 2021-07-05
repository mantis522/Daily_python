def OR_gate(bias, x1, x2, out):
    w1=0
    w2=0
    b=0
    result = b*bias + x1*w1 + x2*w2
    if result <= 0:
        print(b*bias, x1*w1, x2*w2)
        return 0
    else:
        print(b*bias, x1*w1, x2*w2)
        return 1


a = OR_gate(0, 0, 0)

print(a)