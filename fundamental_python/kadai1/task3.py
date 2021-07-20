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

sea_colors = [(0, 15, 93)]
four_colors = [(0, 246, 2), (249, 0, 2), (243, 247, 2), (243, 181, 226)]

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

map_coloring(image, P[1], four_colors[0])
map_coloring(image, P[2], four_colors[1])
map_coloring(image, P[3], four_colors[2])
map_coloring(image, P[4], four_colors[3])
map_coloring(image, P[5], four_colors[2])
map_coloring(image, P[6], four_colors[1])
map_coloring(image, P[7], four_colors[2])
map_coloring(image, P[8], four_colors[3])
map_coloring(image, P[9], four_colors[0])
map_coloring(image, P[10], four_colors[3])
map_coloring(image, P[11], four_colors[0])
map_coloring(image, P[12], four_colors[1])
map_coloring(image, P[13], four_colors[2])
map_coloring(image, P[14], four_colors[3])
map_coloring(image, P[15], four_colors[0])
map_coloring(image, P[16], four_colors[1])
map_coloring(image, P[17], four_colors[2])
map_coloring(image, P[18], four_colors[2])
map_coloring(image, P[17], four_colors[2])
map_coloring(image, P[18], four_colors[2])
map_coloring(image, P[19], four_colors[0])
map_coloring(image, P[20], four_colors[0])
map_coloring(image, P[21], four_colors[2])
map_coloring(image, P[22], four_colors[3])
map_coloring(image, P[23], four_colors[1])
map_coloring(image, P[24], four_colors[3])
map_coloring(image, P[25], four_colors[1])
map_coloring(image, P[26], four_colors[3])
map_coloring(image, P[27], four_colors[0])
map_coloring(image, P[28], four_colors[2])
map_coloring(image, P[29], four_colors[3])
map_coloring(image, P[30], four_colors[1])
map_coloring(image, P[31], four_colors[2])
map_coloring(image, P[32], four_colors[3])
map_coloring(image, P[33], four_colors[1])
map_coloring(image, P[34], four_colors[0])
map_coloring(image, P[35], four_colors[0])
map_coloring(image, P[36], four_colors[1])
map_coloring(image, P[37], four_colors[3])
map_coloring(image, P[38], four_colors[1])
map_coloring(image, P[39], four_colors[0])
map_coloring(image, P[40], four_colors[3])
map_coloring(image, P[41], four_colors[0])
map_coloring(image, P[42], four_colors[0])
map_coloring(image, P[43], four_colors[2])
map_coloring(image, P[44], four_colors[2])
map_coloring(image, P[45], four_colors[2])
map_coloring(image, P[46], four_colors[1])
map_coloring(image, P[47], four_colors[3])
map_coloring(image, P[48], four_colors[1])
map_coloring(image, P[49], four_colors[1])
map_coloring(image, P[50], four_colors[0])
map_coloring(image, P[51], four_colors[3])
map_coloring(image, P[52], four_colors[2])
map_coloring(image, P[53], four_colors[3])
map_coloring(image, P[54], four_colors[2])
map_coloring(image, P[55], four_colors[3])
map_coloring(image, P[56], four_colors[0])
map_coloring(image, P[57], four_colors[1])
map_coloring(image, P[58], four_colors[3])
map_coloring(image, P[59], four_colors[1])
map_coloring(image, P[60], four_colors[3])
map_coloring(image, P[61], four_colors[1])
map_coloring(image, P[62], four_colors[3])
map_coloring(image, P[63], four_colors[2])
map_coloring(image, P[64], four_colors[3])
map_coloring(image, P[65], four_colors[0])
map_coloring(image, P[66], four_colors[0])
map_coloring(image, P[67], four_colors[3])
map_coloring(image, P[68], four_colors[0])
map_coloring(image, P[69], four_colors[1])
map_coloring(image, P[70], four_colors[2])
map_coloring(image, P[71], four_colors[1])
map_coloring(image, P[72], four_colors[2])
map_coloring(image, P[73], four_colors[1])
map_coloring(image, P[74], four_colors[0])
map_coloring(image, P[75], four_colors[1])
map_coloring(image, P[76], four_colors[0])
map_coloring(image, P[77], four_colors[3])
map_coloring(image, P[78], four_colors[0])
map_coloring(image, P[79], four_colors[1])
map_coloring(image, P[80], four_colors[3])
map_coloring(image, P[81], four_colors[2])
map_coloring(image, P[82], four_colors[2])
map_coloring(image, P[83], four_colors[0])
map_coloring(image, P[84], four_colors[0])
map_coloring(image, P[85], four_colors[2])
map_coloring(image, P[86], four_colors[3])
map_coloring(image, P[87], four_colors[3])
map_coloring(image, P[88], four_colors[1])
map_coloring(image, P[89], four_colors[0])
map_coloring(image, P[90], four_colors[1])
map_coloring(image, P[91], four_colors[2])
map_coloring(image, P[92], four_colors[0])
map_coloring(image, P[93], four_colors[1])
map_coloring(image, P[94], four_colors[3])
map_coloring(image, P[95], four_colors[0])
map_coloring(image, P[96], four_colors[1])
map_coloring(image, P[97], four_colors[2])
map_coloring(image, P[98], four_colors[3])
map_coloring(image, P[99], four_colors[0])
map_coloring(image, P[100], four_colors[3])
map_coloring(image, P[101], four_colors[2])
map_coloring(image, P[102], four_colors[0])
map_coloring(image, P[103], four_colors[1])
map_coloring(image, P[104], four_colors[2])
map_coloring(image, P[105], four_colors[3])

map_coloring(image, P[0], sea_colors[0])

image.save("c.png")

print()