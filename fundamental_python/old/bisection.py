# def f(x):
#     return x * x - 2

# a = 0
# b = 3
# def bisection(a, b):
#     epsilon = 0.000001
#     while True:
#         c = (a + b) / 2
#         fc = f(c)
#         if -epsilon <= fc and fc <= epsilon:
#             return c
#         elif f(a) * f(c) < 0:
#             (a, b) = (a, c)
#         else:
#             (a, b) = (c, b)

# print(bisection(0, 3))

# def bisection1(a, b):
#     e = 0.0000001
#     c = (a + b) / 2
#     if abs(a - b) < e:
#         return (a + b) / 2
#     if f(a) * f(c) <= 0:
#         b = c
#     else:
#         a = c
#     return bisection1(a, b)

def f(x, z):
    return x * x - z

def bisection(a, b, z):
    e = 0.0000001
    c = (a + b) / 2
    if abs(a - b) < e:
        return (a + b) / 2
    if f(a, z) * f(c, z) <= 0:
        return bisection(a, c, z)
    else:
        return bisection(c, b, z)

def root(z):
    if z < 1:
        return bisection(0, z+1, z)
    else:
        return bisection(0, z, z)

print(root(4))