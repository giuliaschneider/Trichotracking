import cv2
import numpy as np


def getIntLightBlurred(img, bw, background, w_int):
    """ Returns the weighted intensity matrix ."""
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    intensities = cv2.subtract(cv2.bitwise_not(blurred),
                               cv2.bitwise_not(background))
    # Normalize intensities
    minI = np.quantile(intensities[bw == 255], 0.01)
    maxI = np.quantile(intensities[bw == 255], 0.99)
    intensities[intensities < minI] = minI
    intensities[intensities > maxI] = maxI
    intensities = (intensities - minI) / (maxI - minI)
    intensities = intensities * w_int
    return intensities


def getIntLight(img, bw, background, w_int):
    """ Returns the weighted intensity matrix ."""
    intensities = cv2.subtract(cv2.bitwise_not(img),
                               cv2.bitwise_not(background))
    # Normalize intensities
    minI = np.quantile(intensities[bw == 255], 0.01)
    maxI = np.quantile(intensities[bw == 255], 0.99)
    intensities[intensities < minI] = minI
    intensities[intensities > maxI] = maxI
    intensities = (intensities - minI) / (maxI - minI)
    intensities = intensities * w_int
    return intensities


def getIntDark(img, bw, background, w_int):
    """ Returns the weighted intensity matrix ."""
    intensities = cv2.subtract((img), (background))
    # Normalize intensities
    try:
        minI = np.quantile(intensities[bw == 255], 0.01)
        maxI = np.quantile(intensities[bw == 255], 0.99)
    except:
        set_trace()
    intensities[intensities < minI] = minI
    intensities[intensities > maxI] = maxI
    intensities = (intensities - minI) / (maxI - minI)
    intensities = intensities * w_int
    return intensities
