from PIL import Image, ImageDraw
import sys

im = Image.open("japan.png")

def product(*args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def Counter(word):
    counter = {}
    for letter in word:
        if letter not in counter:
            counter[letter] = 0
        counter[letter] += 1
    return counter

cc = sorted([(v, k) for k, v in Counter(im.getdata()).items()])[-1][1]


for y, x in product(range(im.height), range(im.width)):
    R, G, B = im.getpixel((x, y))[:3]
    if (R, G) == (0, 1):
        im.putpixel(0, 0, B)

n = 0
for y, x in product(range(im.height), range(im.width)):
    if im.getpixel((x, y)) != cc:
        continue
    ImageDraw.floodfill(im, (x, y), (0, 1, n))
    n += 1

dd = [(-1, 0), (0, -1), (0, 1), (1, 0)]
for h in range(1):
    l = list(product(range(1, im.height-1), range(1, im.width-1)))
    for y, x in l:
        c = im.getpixel((x, y))
        if c[:2] == (0, 1):
            for i, j in dd:
                if im.getpixel((x+i, y+j))[:2] != (0, 1):
                    im.putpixel((x+i, y+j), c)


def add_edge(E, x, y):
  if x != y and not (x, y) in E and not (y, x) in E:
    E.append((x,y))

E = []
for y, x in product(range(im.height-1), range(im.width-1)):
    c1 = im.getpixel((x, y))
    if c1[:2] != (0, 1):
        continue
    c2 = im.getpixel((x+1, y))
    c3 = im.getpixel((x, y+1))
    if c2[:2] == (0, 1) and c1[2] != c2[2]:
        add_edge(E, c1[2], c2[2])
    if c3[:2] == (0, 1) and c1[2] != c3[2]:
        add_edge(E, c1[2], c3[2])

count_list = []
for a in range(106):
    count_list.append(a)


## -------------------------------------------------------------------------

#
# r4 = range(4)
# m = LpProblem()
#
# v = [[LpVariable('v%d_%d'%(i, j), cat=LpBinary) for j in r4] for i in count_list]
#
# for i, j in E:
#     for k in r4:
#         m += v[i][k] + v[j][k] <= 1 # (3)
#
#
#
# m.solve()
#
#
# co = [(97, 132, 219), (228, 128, 109), (255, 241, 164), (121, 201, 164)] # 4色
# rr = [int(value(lpDot(r4, i))) for i in v] # 結果
#
# print(rr)
#
# # rr = [3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 0, 2, 2, 0, 1, 1, 2, 1, 3, 0, 0, 3, 0, 1, 2, 0, 1, 1, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 0, 2, 2, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0]
#
# for y, x in product(range(im.height-1), range(im.width-1)):
#     c = im.getpixel((x, y))
#     if c[:2] == (0, 1): # エリアならば、結果で塗る
#         ImageDraw.floodfill(im, (x, y), co[rr[c[2]]])
# im.save('result.png')