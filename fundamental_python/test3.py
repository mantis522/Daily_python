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

image = Image.open("japan.png")
(P, E) = graph(image)


pixel_list = [(1373, 97), (982, 131), (931, 139), (942, 155), (1231, 196), (1275, 245), (1238, 275), (1214, 289), (852, 405), (850, 406),
              (941, 460), (932, 543), (984, 543), (883, 652), (984, 659), (863, 699), (794, 724), (909, 746), (719, 786), (884, 813), (793, 825),
              (832, 826), (931, 830), (698, 831), (716, 872), (462, 880), (839, 886), (651, 889), (445, 896), (442, 897), (889, 910), (790, 913),
              (832, 918), (580, 930), (842, 937), (539, 938), (651, 938), (777, 939), (445, 940), (525, 942), (694, 959), (482, 961), (665, 970),
              (431, 977), (594, 988), (205, 998), (859, 1006), (348, 1010), (628, 1011), (568, 1024), (520, 1025), (189, 1028), (443, 1038),
              (506, 1038), (610, 1039), (447, 1040), (444, 1041), (441, 1043), (438, 1044), (406, 1050), (537, 1051), (434, 1059), (282, 1065),
              (218, 1066), (384, 1068), (477, 1078), (339, 1089), (227, 1095), (203, 1105), (202, 1106), (208, 1106), (201, 1107),
              (204, 1109), (383, 1109), (379, 1110), (378, 1111), (377, 1112), (374, 1114), (299, 1126), (171, 1130), (147, 1148),
              (315, 1153), (258, 1169), (237, 1173), (253, 1176), (237, 1201), (264, 1204), (207, 1236), (1303, 1291), (291, 1316), (1193, 1330),
              (250, 1343), (1086, 1455), (1012, 1477), (1011, 1478), (1010, 1479), (1009, 1480), (1008, 1482), (1007, 1483), (974, 1488), (184, 1498),
              (205, 1511), (204, 1512), (149, 1521), (152, 1524)]

for a in pixel_list:
    P.append(a)

four_colors = [(0, 246, 2), (249, 0, 2), (243, 247, 2), (0, 15, 93)]

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

count = 0

r = [3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 0, 2, 2, 0, 1, 1, 2, 1, 3, 0, 0, 3, 0, 1, 2, 0, 1, 1, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 0, 2, 2, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0]

for a in range(106):
    if count == 4:
        count = 0

    map_coloring(image, P[a], four_colors[r[a]])
    count += 1


image.save("b.png")