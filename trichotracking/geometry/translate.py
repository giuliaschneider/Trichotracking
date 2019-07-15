import numpy as np
import cv2

def translateObjectUp(img, angle, L):
    """ Returns translated image in pos x, y direction. """

    # Translation in positive x, y -direction
    tx = L * np.sin(np.deg2rad(angle))
    ty = -L * np.cos(np.deg2rad(angle))
    T = np.float32([[1, 0, tx], [0, 1, ty]])
    height, width = img.shape[:2]
    up = cv2.warpAffine(img, T, (width, height))
    return up


def translateObjectDown(img, angle, L):
    """ Returns translated image in neg x, y direction. """
    # Translation in negative x, y -direction
    tx = -L * np.sin(np.deg2rad(angle))
    ty = L * np.cos(np.deg2rad(angle))
    T = np.float32([[1, 0, tx], [0, 1, ty]])
    height, width = img.shape[:2]
    down = cv2.warpAffine(img, T, (width, height))
    return down


def translateObject(img, angle, L):
    """ Returns translated image in pos/neg x, y direction. """
    # Translation in positive x, y -direction
    up = translateObjectUp(img, angle, L)
    # Translation in negative x, y -direction
    down = translateObjectDown(img, angle, L)
    return up, down
