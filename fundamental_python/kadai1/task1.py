from reportlib import search, graph, color
from PIL import Image
import sys

def propagate(X, CNF):
    done = False
    while not done:
        done = True
        for (P, N) in CNF:
            I = [ i for i in P if X[i] != False ] + \
                [ i for i in N if X[i] != True ]
            if I == []:
                return False
            i = I.pop()
            if I == [] and X[i] == None:
                X[i] = i in P
                done = False
    return True

def search(X, CNF):
    consistent = propagate(X, CNF)
    if not consistent:
        return None
    if not None in X:
        return X
    i = X.index(None)
    Y = X[:]
    Y[i] = True
    Z = search(Y, CNF)
    if Z != None:
        return Z
    X[i] = False
    return search(X, CNF)

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

file_path = sys.argv[1]

image = Image.open(file_path)
(P, E) = graph(image)

colors = [
  (0, 15, 93), (64,255,64), (255,64,64), (255,255,64),
  (64,255,255), (255,64,255)
]

def color(k):
  global colors
  for i in range(len(colors), k + 1):
    while True:
      r = (255, 0, 0)
      g = (0, 255, 0)
      b = (0, 0, 255)
      c = (r, g, b)
      if not c in colors:
        colors.append(c)
        break
  return colors[k]

def making_color(a, b):
    color_list = []
    if isinstance(b, tuple):
        for i in range(0, len(b)):
            color_list.append(abs(a[i] - b[i]))
        return sum(color_list)

    else:
        return abs(a - b)

def map_coloring(image, coordinate, color):
    next = None
    pic = image.load()
    a, b = coordinate
    background = pic[a, b]
    pic[a, b] = color
    graphs = {(a, b)}

    graphs_set = set()

    while graphs:
        new_graphs = set()
        for (a, b) in graphs:
            for (i, j) in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if (i, j) in graphs_set or i < 0 or j < 0:
                    continue
                try:
                    p = pic[i, j]
                except (ValueError, IndexError):
                    pass
                else:
                    graphs_set.add((i, j))
                    if next is None:
                        filling = making_color(p, background) <= 0
                    else:
                        filling = p != color and p != next
                    if filling:
                        pic[i, j] = color
                        new_graphs.add((i, j))
        graphs_set = graphs
        graphs = new_graphs

map_coloring(image, P[0], color(0))

image.save("a.png")