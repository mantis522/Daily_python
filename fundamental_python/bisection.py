def f(x):
    return x * x - 2

a = 0
b = 3
def bisection(a, b):
    epsilon = 0.000001
    while True:
        c = (a + b) / 2
        fc = f(c)
        if -epsilon <= fc and fc <= epsilon:
            return c
        elif f(a) * f(c) < 0:
            (a, b) = (a, c)
        else:
            (a, b) = (c, b)

print(bisection(0, 3))