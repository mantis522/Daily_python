def inpro(u, v): #innerproduct
    y = 0
    for i in range(0, len(u)):
        y = y + u[i]*v[i]
    return y
def mul(M1, M2):
    m = len(M1)
    n = len(M2)
    Mmul = []
    for i in range(0, m):
        u = M1[i]
        Mmul_i = []
        for j in range(0, m):
            v = []
            for k in range(0, n):
                v = v + [M2[k][j]]
            Mmul_i = Mmul_i + [inpro(u, v)]
        Mmul = Mmul + [Mmul_i]
    return Mmul

A = [[1,2,3],
     [4,5,6]]
B = [[10,20],
     [30,40],
     [50,60]]

a = mul(A, B)
print(a)