import numpy as np
import cv2
import matplotlib.pyplot as plt

from iofiles import loadImage, find_im

from IPython.core.debugger import set_trace


def getChamber(inputDir, background, chamber_function):
    """
    Import or calculate & save roi of image sequence.

    Parameters
    ----------
    inputDir : string
        input directory containing all image files
    background : ndarray
        Median background image

    roi_function : function
        Function that segments roi

    Returns
    -------
    chamber : ndarray
        Roi image

    """

    if "chamber.tif" in os.listdir(inputDir):
        chamber = loadImage(os.path.join(inputDir,"chamber.tif"))
    else:
        chamber = chamber_function(background)
        cv2.imwrite(os.path.join(inputDir, "chamber.tif"), chamber)
    return chamber


def calc_chamber(background):
    """ Get bw image, pixels inside set to 255, outside 0. """
    bw = cv2.adaptiveThreshold(background, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 41, 8)
    # Close gaps by morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    h, w = bw.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(bw, mask, (int(w/2),int(h/2)), 255)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    return bw


def calc_chamber_df_ulisetup(background):
    """ Get bw image, pixels inside set to 255, outside 0,
        adjusted for uli's setup """
    # Find well wall (high intensity circle)
    ret, bw = cv2.threshold(background,200,255,cv2.THRESH_BINARY)
    h, w = bw.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(bw, mask, (int(w/2),int(h/2)), 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    return mask


def dilate_border(chamber, ksize=40):
    chamber[:1,:] = [0]; chamber[-1:,:] = [0]
    chamber[:,:1] = [0]; chamber[:,-1:] = [0]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dchamber = cv2.erode(chamber, kernel)
    return dchamber
