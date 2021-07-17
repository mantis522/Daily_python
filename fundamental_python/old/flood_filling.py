from PIL import Image
import random

# im = Image.open('./org-s.bmp')
im = Image.open('org.jpg')


def isVaildColor(color):
    return color != (102, 102, 102) and color != (153, 204, 255)


def randMapColor():
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    while (not isVaildColor(color)):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color


def findAncestors(colFather):
    while (colorFatherDir[colFather] != colFather):
        colFather = colorFatherDir[colFather]
    return colFather


def MapColorP(color):
    r = color[0]
    g = color[1]
    b = color[2] + 1
    if b > 255:
        b = 0
        g = g + 1
        if g > 255:
            g = 0
            r = r + 1
            print("레드")
            if r > 255:
                print("색상불충분")
                r = 0
    color = (r, g, b)

    if isVaildColor(color):
        return color
    else:
        return MapColorP(color, 1)


print("-------------------------------처리중----------------------------------")

color = (0, 0, 0)
color = MapColorP(color)

colorFatherDir = {}
colorFatherDir[color] = color
for i in range(im.size[0]):
    if isVaildColor(im.getpixel((i, 0))):
        im.putpixel((i, 0), color)
    else:
        color = MapColorP(color)
        colorFatherDir[color] = color

for j in range(1, im.size[1]):
    if isVaildColor(im.getpixel((0, j))):
        if isVaildColor(im.getpixel((0, j - 1))):
            colorMate = im.getpixel((0, j - 1))
            im.putpixel((0, j), colorMate)
        else:
            color = MapColorP(color)
            colorFatherDir[color] = color
            im.putpixel((0, j), color)
    for i in range(1, im.size[0]):
        if isVaildColor(im.getpixel((i, j))):
            if isVaildColor(im.getpixel((i, j - 1))):
                colorMate = im.getpixel((i, j - 1))
                im.putpixel((i, j), colorMate)

                if isVaildColor(im.getpixel((i - 1, j))):
                    if findAncestors(im.getpixel((i, j))) != findAncestors(im.getpixel((i - 1, j))):
                        colorFatherDir[findAncestors(im.getpixel((i, j)))] = findAncestors(im.getpixel((i - 1, j)))
            elif isVaildColor(im.getpixel((i - 1, j))):
                colorMate = im.getpixel((i - 1, j))
                im.putpixel((i, j), colorMate)
            else:
                color = MapColorP(color)
                colorFatherDir[color] = color
                im.putpixel((i, j), color)

print("총 색상 수", len(colorFatherDir))
realColor = {}
for father in colorFatherDir:
    realColor[findAncestors(father)] = randMapColor()

for j in range(1, im.size[1]):
    for i in range(1, im.size[0]):
        if isVaildColor(im.getpixel((i, j))):
            colFather = im.getpixel((i, j))
            im.putpixel((i, j), realColor[findAncestors(colFather)])

im.show()
im.save("tmp.bmp", "BMP")