import glob
import numpy as np
import cv2
from ImageLoader import imageload as loader

images = [loader.cv_imread(file) for file in glob.glob('./images/*')]

for img in images:
    cv2.imshow('image', loader.image_resize(img, height = 600))
    key = cv2.waitKey()
    if key == ord('q'):
        break
    else:
        continue

cv2.destroyAllWindows()
