import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    #cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img


def image_resize(image, width = None, height = None, inter = cv2.INTER_CUBIC):
    dim = None
    (h, w) = image.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def image_text(img, text, pos, font_path = 'fonts/simsun.ttc', size = 26, color = (255, 255, 255)):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    fontText = ImageFont.truetype(font_path, size, encoding="utf-8")
    draw.text(pos, text, color, font=fontText)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)



def load_all(imageFolder):
    images = [cv_imread(file) for file in glob.glob(imageFolder+'/*')]
    return images