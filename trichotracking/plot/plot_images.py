import matplotlib.pyplot as plt
import cv2

from ._draw import drawRect


__all__ = ['plotAllImages', 'plotAllContoursTracking', 'plotAllContours']


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

