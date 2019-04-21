import cv2
import glob
import numpy as np


files = [file for file in glob.glob('./images/*')]
for file in files:
    # image = cv2.imdecode(np.fromfile(file), -1)
    image = cv2.imread(file)
    cv2.imshow('image', image)

    key = cv2.waitKey()
    if key == ord('q'):
        break
    else:
        continue

cv2.destroyAllWindows()
