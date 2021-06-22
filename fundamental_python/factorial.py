def factorial(n):
    fac = 1
    for a in range(1, n+1):
        fac *= a
    return fac

b = factorial(5)

print(b)