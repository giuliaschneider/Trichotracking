import os
import os.path

import cv2

from trichotracking.iofiles import loadImage


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
        chamber, h, w = loadImage(os.path.join(inputDir, "chamber.tif"))
    else:
        chamber = chamber_function(background)
        cv2.imwrite(os.path.join(inputDir, "chamber.tif"), chamber)
    return chamber


def dilate_border(chamber, ksize=40):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    chamber = cv2.morphologyEx(chamber, cv2.MORPH_CLOSE, kernel)
    chamber[:1, :] = [0];
    chamber[-1:, :] = [0]
    chamber[:, :1] = [0];
    chamber[:, -1:] = [0]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dchamber = cv2.erode(chamber, kernel)
    return dchamber
