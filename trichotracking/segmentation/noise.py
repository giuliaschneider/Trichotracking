import numpy as np
import cv2

__all__ = ['removeNoise']


def removeNoise(bw, minArea):
    """ Removes all objects smaller than minArea. """
    nObjects, labelledImage, stats, centroids = \
        cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
    objectAreas = stats[:,cv2.CC_STAT_AREA]
    labels = np.arange(nObjects)
    noiseLabels = labels[objectAreas<minArea]
    for i in noiseLabels:
        bw[labelledImage==i] = 0
    return bw