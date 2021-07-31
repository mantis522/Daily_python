import math

def cal(x, y):
    return x ** y

# a = cal(0.6, 5) * cal(0.4, 5)
# b = cal(0.5, 5) * cal(0.5, 5)
#
# c = a / (a+b)
# print(c)

def gaussian(a, b, c):
    return 1 / (math.sqrt(2 * 3.14) * b) * math.exp((-(a-c)**2) / 2 * b**2)

a = gaussian(6, 1.0, 6)
print(round(a, 5))

b = gaussian(2, 1.0, 6)
print(round(b, 5))

c = gaussian(4, 1.0, 6)
print(round(c, 5))

z = 1/3 * a + 1/3 * b + 1/3 * c
print(z)

p_a = 1/3 * a / z
print(p_a)

p_b = 1/3 * b / z
print(p_b)

p_c = 1/3 * c / z
print(p_c)