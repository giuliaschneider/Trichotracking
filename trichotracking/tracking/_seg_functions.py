import cv2
from iofiles import loadImage
from plot.plot_images import plotAllImages

from IPython.core.debugger import set_trace



def segment_particles(inimg, background, chamber=None, blur=True, 
                      darkField=False, plotImages=False, threshold=28):
    """ Segments image and returns image and semgmented image. """
    img = inimg
    if blur:
        img = cv2.GaussianBlur(img,(5, 5),0)
    if darkField:
        subtracted = cv2.subtract((img), (background))
        #bw = cv2.adaptiveThreshold(cv2.bitwise_not(subtracted),255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,13,6)
        _, bw = cv2.threshold(subtracted, 29, 255, cv2.THRESH_BINARY)

    else:
        subtracted = cv2.subtract(cv2.bitwise_not(img),
                                  cv2.bitwise_not(background))
        _, bw = cv2.threshold(subtracted, threshold, 255, cv2.THRESH_BINARY)

    # Close gaps by morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    if chamber is not None:
        bw[chamber == 0] = [0]

    # plot images
    if plotImages:
        imgs = [img, bw, subtracted]
        labels = ["Original", "Threshold", "Sub"]
        plotAllImages(imgs, labels)
    return img, bw


