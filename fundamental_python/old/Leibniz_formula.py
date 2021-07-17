def leibniz(n):
    sign, pi = 1, 0.0
    for a in range(n):
        pi += 4 / (a*2+1) * sign
        sign *= -1
    return pi

b = leibniz(1000)
print(b)