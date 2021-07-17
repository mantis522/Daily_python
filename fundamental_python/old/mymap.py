def mymap(f, L):
    M = []
    for x in L:
        print(f(x))
        M = M + [f(x)]
    return M

def double(x):
    return x * 2



def mymap1(f, L):
    M = []
    for x in L:
        M.append(f(x))
    return M

a = mymap1(double, [1, 2, 3])
print(a)

def concat(L):
    sum = 0
    for x in L:
        sum = sum + x
    return sum

b = concat([1, 2, 3])
print(b)

def product(A, B):
    Q = []
    for a in A:
        for b in B:
            M = []
            M.append(a)
            M.append(b)
            Q.append(tuple(M))
    return Q

c = product([1, 2, 3], ['a', 'b'])
print(c)

def reverse(L):
    M = []
    for a in range(len(L)):
        M.append(L.pop())
    return M

print(reverse([1, 2, 3]))


def bsort(L):
    list_length = len(L)
    for i in range(list_length-1):
        for j in range(list_length-i-1):
            if L[j] > L[j+1]:
                L[j], L[j+1] = L[j+1], L[j]
    return L

d = bsort([1, 3, 4, 2])
print(d)