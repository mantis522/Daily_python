# import networkx as nx
from pulp import *
from PIL import Image, ImageDraw
from collections import Mapping, Set

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

def graph(image):
  (w, h) = image.size
  V = []
  E = []
  P = {}
  M = []
  for y in range(0, h):
    L = []
    for x in range(0, w):
      c = image.getpixel((x, y))
      if c != (0, 0, 0) and not c in P:
        P[c] = len(V)
        V.append((x, y))
      if c == (0, 0, 0):
        L.append(None)
      else:
        L.append(P[c])
    M.append(L)
  for y in range(0, h):
    for x in range(0, w):
      u = M[y][x]
      if u != None:
        for (x2, y2) in [(x + 3, y), (x, y + 3)]:
          if x2 < w and y2 < h:
            v = M[y2][x2]
            if v != None:
              add_edge(E, u, v)
  return (V, E)