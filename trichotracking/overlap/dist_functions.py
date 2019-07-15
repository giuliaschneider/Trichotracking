import numpy as np
import cv2


def getDist(bw_filled, w_dist):
    """ Returns the weighted distance matrix to the object boundary."""
    h, w = bw_filled.shape[:2]
    bwpadded = np.zeros((h+2, w+2)).astype(np.uint8)
    bwpadded[1:-1,1:-1] = bw_filled
    dist = cv2.distanceTransform(bwpadded, cv2.DIST_L2, 3)
    dist = dist[1:-1,1:-1]
    dist = dist * w_dist
    return dist
