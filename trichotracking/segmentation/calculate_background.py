import os
from os.path import join
import numpy as np
import cv2

from iofiles import loadImage, find_im


def getBackground(inputDir, blur=True, notMoving=False):
    """
    Import or calculate & saves median image of image sequence.

    Parameters
    ----------
    inputDir : string
        input directory containing all image files

    blur : bool, optional
        If True, blurs images with Gaussian blur, wsize of 5

    Returns
    -------
    background : ndarray
        Median image

    """

    # Check if background is already
    if "background.tif" in os.listdir(inputDir):
        background, h, w = loadImage(join(inputDir,"background.tif"))
        return background


    images = find_img(inputDir)
    background = calcBackground(images, blurBg, notMoving)
    cv2.imwrite(os.path.join(inputDir, "background.tif"), background)
    return background



def calcBackground(images, blur=True):
    """
    Calculates median image of every 10th image.

    Parameters
    ----------
    images : list of strings
        list containing all image file paths

    blur : bool, optional
        If True, blurs images with Gaussian blur, wsize of 5

    Returns
    -------
    background : ndarray
        Median image
    """

    nFiles = len(images)
    img, height, width = loadImage(images[0])
    files = np.arange(0, nFiles, int(0.1*nFiles))

    # Calculate the median image
    imgs = np.zeros((height,width,files.size), np.uint8)
    for i, index in enumerate(files):
        filepath = images[index]
        img, height, width = loadImage(filepath)
        if blur:
            img = cv2.GaussianBlur(img,(5, 5),0)
        imgs[:,:,i] = img
    bg = np.uint8(np.median(imgs, axis=2))
    return bg
