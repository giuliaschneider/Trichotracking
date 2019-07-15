import numpy as np
from scipy.spatial.distance import euclidean


def minDistBondingBox(box1, box2):
    """ Returns the minimal distance between two upright bounding boxes.

        https://stackoverflow.com/questions/4978323/how-to-calculate-
        distance-between-two-rectangles-context-a-game-in-lua#26178015"""

    x1, y1, bw1, bh1 = box1[0], box1[1], box1[2], box1[3]
    x1b, y1b = x1+bw1, y1+bh1
    x2, y2, bw2, bh2 = box2[0], box2[1], box2[2], box2[3]
    x2b, y2b = x2+bw2, y2+bh2
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return euclidean([x1, y1b], [x2b, y2])
    elif left and bottom:
        return euclidean([x1, y1], [x2b, y2b])
    elif bottom and right:
        return euclidean([x1b, y1], [x2, y2b])
    elif right and top:
        return euclidean([x1b, y1b], [x2, y2])
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:             # rectangles intersect
        return 0.


def minDistBondingBoxes(boxes1, boxes2):

    dist = np.empty((boxes1.shape[0], boxes2.shape[0]))
    for box1 in range(boxes1.shape[0]):
        for box2 in range(boxes2.shape[0]):
            dist[box1, box2] = minDistBondingBox(boxes1[box1,:],
                                                 boxes2[box2,:])
    return dist
