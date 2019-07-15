import numpy as np
import cv2
from iofiles import loadImage


def calcBackgroundNotMoving(listOfFiles):
    """ Calculates the static background, save to self.background."""
    nFiles = len(listOfFiles)
    img, height, width = loadImage(listOfFiles[0])
    nbgs = np.zeros((height,width,10), np.uint8)
    step = int(nFiles/10)
    startInd = 0

    for i in range(10):
        endInd = min(startInd + step, nFiles)
        nbgs[...,i] = calcBlurredBackground(listOfFiles,
                        startFile=startInd, endFile=endInd)
        startInd = endInd + step

    background = np.uint8(np.max(nbgs, axis=2))
    return background


def calcBackground(listOfFiles):
    """ Calculates the static background, save to self.background."""
    nFiles = len(listOfFiles)
    img, height, width = loadImage(listOfFiles[0])
    # Calculate the median image
    step = int(0.1*nFiles)
    files = np.arange(0, nFiles, step)
    imgs = np.zeros((height,width,files.size), np.uint8)
    for i, index in enumerate(files):
        filepath = listOfFiles[index]
        img, height, width = loadImage(filepath)
        imgs[:,:,i] = img
    background1 = np.uint8(np.median(imgs, axis=2))
    return background1


def calcBlurredBackground(listOfFiles, startFile=None, endFile=None):
    """ Calculates the static background, save to self.background."""
    nFiles = len(listOfFiles)
    img, height, width = loadImage(listOfFiles[0])
    # Calculate the median image
    if startFile is None:
        startFile = 0
    if endFile is None:
        endFile = nFiles
    step = int(0.1*nFiles)
    #files = np.arange(nFiles-150, nFiles, step)
    files = np.arange(startFile, endFile, step)
    imgs = np.zeros((height,width,files.size), np.uint8)
    for i, index in enumerate(files):
        filepath = listOfFiles[index]
        img, height, width = loadImage(filepath)
        img = cv2.GaussianBlur(img,(7, 7),0)
        imgs[:,:,i] = img
    background1 = np.uint8(np.median(imgs, axis=2))
    return background1
