# name : Kim UngHee
# id : 2110057
# acknowledgements : None

# from reportlib import search, graph, color
from PIL import Image
import sys

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

## reportライブラリーの「color」を使えば、下のall_colorsリストを使わなくてもいいが、report-4.pdfにはライブラリーのファイル名が「report.py」ですが、講義資料があるウェブサイトにはライブラリーのファイル名が「reportlib.py」なのでエラーが出る場合に備え、以下のようにリストを作成しました。
## reportライブラリーの「color」を使用するコードは154,155ラインに注釈処理をしておきました。

all_colors = [(193, 239, 66), (142, 72, 85), (213, 76, 107), (61, 241, 73), (191, 64, 15), (183, 182, 36),
              (171, 79, 10), (169, 16, 183), (246, 83, 145), (145, 120, 26), (152, 123, 85), (53, 188, 171), (187, 46, 47),
              (52, 98, 41), (104, 136, 225), (21, 112, 60), (254, 117, 252), (199, 2, 102), (45, 226, 26), (60, 150, 27),
              (70, 248, 23), (245, 251, 0), (30, 99, 153), (225, 123, 20), (81, 232, 86), (53, 57, 179), (121, 101, 6),
              (2, 227, 186), (54, 221, 168), (97, 222, 85), (85, 85, 188), (256, 199, 202), (164, 14, 98), (17, 100, 40), (139, 144, 82),
              (90, 99, 132), (113, 10, 199), (86, 31, 73), (16, 228, 44), (229, 26, 126), (38, 135, 52), (167, 235, 65), (29, 254, 18),
              (114, 178, 47), (98, 171, 178), (128, 125, 72), (111, 30, 137), (1, 146, 204), (40, 213, 255), (122, 81, 66), (147, 152, 70),
              (196, 211, 251), (0, 101, 131), (51, 157, 212), (61, 111, 204), (24, 118, 67), (216, 215, 92), (246, 95, 13), (131, 33, 82),
              (23, 201, 180), (143, 111, 172), (149, 175, 232), (179, 6, 146), (223, 37, 143), (118, 246, 234), (168, 118, 68), (189, 16, 219),
              (170, 73, 29), (122, 159, 28), (249, 80, 177), (141, 8, 115), (124, 229, 144), (22, 33, 19), (36, 136, 238), (140, 137, 234),
              (0, 119, 2), (111, 98, 229), (29, 236, 50), (158, 14, 240), (103, 61, 52), (49, 234, 149), (191, 40, 224), (144, 200, 194),
              (17, 239, 135), (165, 204, 48), (79, 36, 205), (173, 64, 44), (82, 155, 255), (29, 198, 130), (226, 57, 107), (132, 120, 35),
              (122, 169, 131), (99, 101, 1), (182, 28, 160), (122, 169, 18), (17, 128, 249), (228, 231, 170), (144, 77, 245), (160, 7, 132),
              (238, 223, 156), (106, 3, 94), (111, 156, 75), (24, 154, 184), (148, 250, 179), (26, 87, 112), (247, 213, 103)]


# for a in range(len(P)):
#     map_coloring(image, P[a], color(a))

for a in range(len(P)):
    map_coloring(image, P[a], all_colors[a])

map_coloring(image, P[0], sea_colors[0])
image.save("a.png")