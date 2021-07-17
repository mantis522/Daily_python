def gcd1(x, y):
    if y == 0:
        return x
    else:
        return gcd1(y, x % y)

def gcd2(x, y):
    while y > 0:
        (x, y) = (y, x % y)
    return x