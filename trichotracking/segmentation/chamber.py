import os
import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from iofiles import loadImage, find_img

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



def dilate_border(chamber, ksize=40):
    chamber[:1,:] = [0]; chamber[-1:,:] = [0]
    chamber[:,:1] = [0]; chamber[:,-1:] = [0]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dchamber = cv2.erode(chamber, kernel)
    return dchamber
