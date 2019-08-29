import numpy as np


def cropRectangle(img, bx, by, bw, bh, mult=1):
    """ Crops the image to the given geometry. """
    dy = int((mult * bh - bh) / 2)
    dx = int((mult * bw - bw) / 2)
    left = max(0, int(bx - dx))
    right = min(img.shape[1], int(bx + bw + dx))
    upper = max(0, int(by - dy))
    lower = min(img.shape[0], int(by + bh + dy))
    return img[upper:lower, left:right], left, upper


def cropRectangleKeepSize(img, bx, by, bw, bh, mult=1):
    """ Crops the image to the given geometry. """
    bx, by = int(bx), int(by)
    bw, bh = int(bw), int(bh)
    cropped, left, upper = cropRectangle(img, bx, by, bw, bh, mult)
    dy = int((mult * bh - bh) / 2)
    dx = int((mult * bw - bw) / 2)
    nBh, nBw = cropped.shape
    m = np.mean(cropped).astype(np.uint8)
    if int(bx - dx) < 0:
        pad = np.zeros((nBh, 0 - (bx - dx))) + m
        cropped = np.hstack((pad, cropped))
        left = (bx - dx)
    nBh, nBw = cropped.shape
    if int(bx + bw + dx) > img.shape[1]:
        pad = np.zeros((nBh, (bx + bw + dx) - img.shape[1])) + m
        cropped = np.hstack((cropped, pad))
    nBh, nBw = cropped.shape
    if int(by - dy) < 0:
        pad = np.zeros((0 - (by - dy), nBw)) + m
        cropped = np.vstack((pad, cropped))
        upper = (by - dy)
    nBh, nBw = cropped.shape
    if int(by + bh + dy) > img.shape[0]:
        pad = np.zeros(((by + bh + dy) - img.shape[0], nBw)) + m
        cropped = np.vstack((cropped, pad))
    cropped = np.uint8(cropped)
    return cropped, left, upper
