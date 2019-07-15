import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as mtransforms
import cv2
import colorsys


def hsv2rgb(h,s,v):
    rgb = ([colorsys.hsv_to_rgb(hi,s,v) for hi in h])
    r = [r[0] for r in rgb ]
    g = [r[1] for r in rgb ]
    b = [r[2] for r in rgb ]
    rgb = np.array([colorsys.hsv_to_rgb(hi,s,v) for hi in h])
    return rgb

def drawRect(img, min_rect, r, g, b):
    for rect in min_rect:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(b,g,r),3)


def plotAllImages(imgs, labels):
    for i, img in enumerate(imgs):
        fig = plt.figure(i+1)
        plt.imshow(img, cmap="gray")
        plt.title(labels[i])

def plotAllContoursTracking(img, contours, cx,cy):
    """ Plots all contours in a matplotlib plot. """
    plt.figure()
    plt.imshow(img, cmap="gray")
    for c, x, y in zip(contours, cx, cy):
        plt.plot(c[:,0,0], c[:,0,1])
        plt.plot(x,y,'o')

def plotAllContours(img, contours, min_rect=None):
    """ Plots all contours in a matplotlib plot. """

    if min_rect is not None:
        img = img.copy()
        drawRect(img, min_rect, 0, 0, 255)


    plt.figure()
    plt.imshow(img, cmap="gray")
    for c in contours:
        plt.plot(c[:,0,0], c[:,0,1])


def saveAllContours(filename, img, contours, min_rect=None):
    """ Plots all contours in a matplotlib plot. """

    img_new = img.copy()

    cimg = cv2.cvtColor(img_new, cv2.COLOR_GRAY2RGB)

    if min_rect is not None:
        for rect in min_rect:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(cimg,[box],0,(0,0,0),2)



    for i, c in enumerate(contours):
        r, g, b = hsv2rgb(i/len(contours), 1, 1)
        cv2.drawContours(cimg, [c], 0, (g,b,r), 2)

    cv2.imwrite(filename, cimg)



def plotFilledArea(filledImage):
    plt.figure()
    plt.imshow(filledImage)


def plotStats(stat1, stat2):
    """Plots the contour area vs the eccentricity """
    plt.figure()
    plt.plot(stat1, stat2,'o')
