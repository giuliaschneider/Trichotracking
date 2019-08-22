import cv2
import numpy as np

__all__ = ['drawRect']


def drawRect(img, min_rect, r, g, b):
    for rect in min_rect:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (b, g, r), 3)
