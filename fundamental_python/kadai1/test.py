from reportlib import search, graph, color
from PIL import Image, ImageDraw

## https://pillow.readthedocs.io/en/stable/_modules/PIL/ImageDraw.html#floodfill

# region i is of color k
def var(i, k):
  return i * 4 + k

image = Image.open("japan-colored.png")
# graph(image) coverts a given image into the corresponding graph.
# Here the colors of regions in the image need to be pairwise distinct.
(P, E) = graph(image)
print(P)

# P.append((932, 543))
# P.append((1373, 97))
# P.append((932, 543))


# CNF = [
#   # every region is painted by exactly one color.
#   ([var(0, 0), var(0, 1), var(0, 2), var(0, 3)], []),
#   ([], [var(0, 0), var(0, 1)]),
#   ([], [var(0, 0), var(0, 2)]),
#   ([], [var(0, 0), var(0, 3)]),
#   ([], [var(0, 1), var(0, 2)]),
#   ([], [var(0, 1), var(0, 3)]),
#   ([], [var(0, 2), var(0, 3)]),
#   ...
#   # for every (i, j) in E the colors of regions i and j must be
#   # different:
#   ([], [var(0, 0), var(1, 0)]),
#   ([], [var(0, 1), var(1, 1)]),
#   ([], [var(0, 2), var(1, 2)]),
#   ([], [var(0, 3), var(1, 3)]),
#   ...
# ]

# 5 regions * 4 colors = 20 variables
# X = search([None] * 20, CNF)
# print(X)

# ImageDraw.floodfill(image, P[0], color(0)) # darkblue
# ImageDraw.floodfill(image, P[1], color(2)) # green
# ImageDraw.floodfill(image, P[2], color(5)) # green
# ImageDraw.floodfill(image, P[3], color(2)) # green
# ImageDraw.floodfill(image, P[4], color(3)) # green

# image.save("a.png")