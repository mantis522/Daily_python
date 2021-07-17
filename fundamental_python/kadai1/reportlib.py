from random import randrange

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

# See l9.pdf.
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

# graph(image) takes an RGB image to return (P, E).  Here P is a list of
# representative positions for all non-black colors --- actually for each
# color the topmost leftmost pixel of the color is chosen; and E is a list
# of edges (i, j) that satisfies
#  - the colors at (x_i,y_i) and (a,b) are same,
#  - the colors at (x_j,y_j) and (c,d) are same, and
#  - (a,b) - (c,d) is either (0,3) or (3,0).
# for some positions (a, b) and (c, d) of non-black pixels in the image.
#
# For a map image if every region is painted by a distinct color,
# graph(image) returns the corresponding graph structure,
# where vertices are 0, 1, ..., n - 1, and edges are listed in L.
# Performing a flood-filling algorithm at P[i], we can paint the i-th
# region.
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

colors = [
  (64,64,128), (64,255,64), (255,64,64), (255,255,64),
  (64,255,255), (255,64,255)
]

def color(k):
  global colors
  for i in range(len(colors), k + 1):
    while True:
      r = randrange(64, 255, 16)
      g = randrange(64, 255, 16)
      b = randrange(64, 255, 16)
      c = (r, g, b)
      if not c in colors:
        colors.append(c)
        break
  return colors[k]