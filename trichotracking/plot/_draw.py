import numpy as np
import cv2


__all__ = ['drawRect']

def drawRect(img, min_rect, r, g, b):
    for rect in min_rect:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(b,g,r),3)