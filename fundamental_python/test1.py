from reportlib import search, graph, color
from PIL import Image, ImageDraw

## https://pillow.readthedocs.io/en/stable/_modules/PIL/ImageDraw.html#floodfill

# region i is of color k
def var(i, k):
  return i * 4 + k

image = Image.open("japan.png")

(P, E) = graph(image)

def floodfill(image, xy, value, border=None, thresh=0):

    pixel = image.load()
    x, y = xy
    try:
        background = pixel[x, y]
        if _color_diff(value, background) <= thresh:
            return  # seed point already has fill color
        pixel[x, y] = value
    except (ValueError, IndexError):
        return  # seed point outside image
    edge = {(x, y)}
    # use a set to keep record of current and previous edge pixels
    # to reduce memory consumption
    full_edge = set()
    while edge:
        new_edge = set()
        for (x, y) in edge:  # 4 adjacent method
            for (s, t) in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                # If already processed, or if a coordinate is negative, skip
                if (s, t) in full_edge or s < 0 or t < 0:
                    continue
                try:
                    p = pixel[s, t]
                except (ValueError, IndexError):
                    pass
                else:
                    full_edge.add((s, t))
                    if border is None:
                        fill = _color_diff(p, background) <= thresh
                    else:
                        fill = p != value and p != border
                    if fill:
                        pixel[s, t] = value
                        new_edge.add((s, t))
        full_edge = edge  # discard pixels processed
        edge = new_edge


def _color_diff(color1, color2):

    if isinstance(color2, tuple):
        return sum([abs(color1[i] - color2[i]) for i in range(0, len(color2))])
    else:
        return abs(color1 - color2)

floodfill(image, P[0], color(0)) # darkblue

image.save("a.png")