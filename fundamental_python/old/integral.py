from math import *

def f(x):
    return x * sin(x)


def integral(f, a, b):
    n = 10000
    dx = (b-a) / n
    y = 0
    for k in range(1, n):
        y = y + f(a + k * dx)
    y = y + (f(a) + f(b)) / 2
    return dx * y

a = 0
b = 2 * 3.14
print(integral(f, a, b))

# def f2(x):
#     return x * x - 2
#
# def derivative(f, x):
#     d = 1e - 10
#     return (f(x+d) - f(x)) / d
#
# def newton(f, x):
#     epsilon = 1e - 10