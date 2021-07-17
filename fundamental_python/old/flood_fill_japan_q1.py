### https://www.geeksforgeeks.org/floodfill-image-using-python-pillow/

import os.path
import sys

from PIL import Image, ImageDraw, PILLOW_VERSION

def get_img_dir() -> str:
    pkg_dir = os.path.dirname(__file__)
    img_dir = os.path.join(pkg_dir)
    return img_dir


if __name__ == '__main__':
    input_img = os.path.join(get_img_dir(), 'japan.png')
    image = Image.open(input_img)
    width, height = image.size
    center = (int(0.7 * width), int(0.7 * height))
    yellow = (154, 125, 14)
    ImageDraw.floodfill(image, xy=center, value=yellow)
    output_img = os.path.join(get_img_dir(), 'japan_test.png')
    image.save(output_img)

    print('Using Python version {}'.format(sys.version))
    print('Using Pillow version {}'.format(PILLOW_VERSION))