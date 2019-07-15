import cv2
from iofiles import loadImage
from plot.plot_images import plotAllImages

from IPython.core.debugger import set_trace


def segementTrichosBlurred(img, background=None, plotImages=False,
                           threshold=28, chamber=None):
    """ Segments image and saves objects to self.contours. """
    blurred = cv2.GaussianBlur(img,(5, 5),0)
    subtracted = cv2.subtract(cv2.bitwise_not(blurred),
                 cv2.bitwise_not(background))
    ret, bw = cv2.threshold(subtracted, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # plot images
    if plotImages:
        imgs = [img, bw, subtracted]
        labels = ["Original", "Threshold", "Sub"]
        plotAllImages(imgs, labels)
    return img, bw


def segementTrichosLight(filepath, background=None, plotImages=False):
    """ Segments image and saves objects to self.contours. """
    subtracted = cv2.subtract(cv2.bitwise_not(img),
                              cv2.bitwise_not(background))
    ret, bw = cv2.threshold(subtracted, 17, 255, cv2.THRESH_BINARY)
    # Close gaps by morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # plot images
    if plotImages:
        imgs = [img, bw, background]
        labels = ["Original", "Threshold", "bg"]
        plotAllImages(imgs, labels)
    return img, bw


def segementTrichosDarkField(img, background=None, plotImages=False,
                             chamber=None, threshold=11):
    """ Segments image and saves objects to self.contours. """
    blurred = cv2.GaussianBlur(img,(5, 5),0)
    substracted = cv2.subtract((blurred), (background))

    if chamber is not None:
        substracted[chamber==0] = 0

    bw = cv2.adaptiveThreshold(cv2.bitwise_not(substracted),255, \
                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,31,4)
    # Close gaps by morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    if plotImages:
        imgs = [img, bw]
        labels = ["Original", "Threshold"]
        plotAllImages(imgs, labels)
    return img, bw
