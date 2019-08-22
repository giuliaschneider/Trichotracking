import numpy as np
import numpy.linalg as la
from scipy.spatial.distance import euclidean


def minDistBoundingBox(box1, box2):
    """ Returns the minimal distance between two upright bounding boxes.

        https://stackoverflow.com/questions/4978323/how-to-calculate-
        distance-between-two-rectangles-context-a-game-in-lua#26178015"""

    x1, y1, bw1, bh1 = box1[0], box1[1], box1[2], box1[3]
    x1b, y1b = x1 + bw1, y1 + bh1
    x2, y2, bw2, bh2 = box2[0], box2[1], box2[2], box2[3]
    x2b, y2b = x2 + bw2, y2 + bh2
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
    else:  # rectangles intersect
        return 0.


def minDistBoundingBoxes(boxes1, boxes2):
    """ Returns a matrix of minimal distances between boxes 1 and 2. """

    dist = np.empty((boxes1.shape[0], boxes2.shape[0]))
    for box1 in range(boxes1.shape[0]):
        for box2 in range(boxes2.shape[0]):
            dist[box1, box2] = minDistBoundingBox(boxes1[box1, :],
                                                  boxes2[box2, :])
    return dist


def do_polygons_intersect(a, b):
    """
    Determines if two polygons a and b intersect.
    Uses the Separating Axis Theorem, Code: https://stackoverflow.com/a/56962827

    Parameters
    ----------
    a : ndarray
        array of connected points [[x1, y1], [x2, y2],...]], form a closed polygon
    b : ndarray
        array of connected points form a closed polygon

    Returns
    -------
    intersect : boolean
        True, if polygons intersect
    """

    polygons = [a, b]
    minA, maxA, projected, i, i1, j, minB, maxB = None, None, None, None, None, None, None, None

    for i in range(len(polygons)):

        # for each polygon, look at each edge of the polygon, and determine if it separates
        # the two shapes
        polygon = polygons[i]
        for i1 in range(len(polygon)):

            # grab 2 vertices to create an edge
            i2 = (i1 + 1) % len(polygon)
            p1 = polygon[i1]
            p2 = polygon[i2]

            # find the line perpendicular to this edge
            normal = {'x': p2[1] - p1[1], 'y': p1[0] - p2[0]}

            minA, maxA = None, None
            # for each vertex in the first shape, project it onto the line perpendicular to the edge
            # and keep track of the min and max of these values
            for j in range(len(a)):
                projected = normal['x'] * a[j][0] + normal['y'] * a[j][1]
                if (minA is None) or (projected < minA):
                    minA = projected

                if (maxA is None) or (projected > maxA):
                    maxA = projected

            # for each vertex in the second shape, project it onto the line perpendicular to the edge
            # and keep track of the min and max of these values
            minB, maxB = None, None
            for j in range(len(b)):
                projected = normal['x'] * b[j][0] + normal['y'] * b[j][1]
                if (minB is None) or (projected < minB):
                    minB = projected

                if (maxB is None) or (projected > maxB):
                    maxB = projected

            # if there is no overlap between the projects, the edge we are looking at separates the two
            # polygons, and we know there is no overlap
            if (maxA < minB) or (maxB < minA):
                return False

    return True


def boxCenter(box):
    """
    Returns center of box

    Parameters
    ----------
    box : ndarray
        array of vertice coordinates in order, i.e. [[0,0], [0,1], [1,1], [1.0]]

    Returns
    -------
    [cx, cy] : array of center coordinates
    """
    return (box[0, :] + box[2, :]) / 2


def boxExtent(box):
    """
    Returns half extent of box

    Parameters
    ----------
    box : ndarray
        array of vertice coordinates in order, i.e. [[0,0], [0,1], [1,1], [1.0]]

    Returns
    -------
    [x-extent, y-extent] : array of extent

    """
    return (np.max(box, axis=0) - np.min(box, axis=0)) / 2


def minDistMinBox(box1, box2):
    """
    Returns the minimal distance between box1 and box2
    Code: https://stackoverflow.com/a/35066128

    Parameters
    ----------
    box1 : list 
        list of vertice coordinates in order, i.e. [[0,0], [0,1], [1,1], [1.0]]
    box2 : list
        list of vertice coordinates in order

    Returns
    -------
    dmin : float
        minimal distance between box1 and box 2

    """

    # Check if boxes are overlapping
    doOverlap = do_polygons_intersect(box1, box2)
    if doOverlap:
        return 0.0

    box1 = np.array(box1)
    box2 = np.array(box2)
    c1 = boxCenter(box1)
    c2 = boxCenter(box2)
    extent1 = boxExtent(box1)
    extent2 = boxExtent(box2)

    dmin = la.norm(np.maximum(0, np.abs(np.abs(c1 - c2) - (extent1 + extent2))))

    return dmin


def minDistMinBoxes(boxes1, boxes2):
    """
    Returns the matrix of minimal distances between boxes1 and boxes2

    Parameters
    ----------
    box1 : list 
        list of vertice coordinates in order, i.e. [[0,0], [0,1], [1,1], [1.0]]
    box2 : list
        list of vertice coordinates in order

    Returns
    -------
    dist : ndarray
        matrix of minimal distance between boxes 1 and 2

    """
    dist = np.empty((boxes1.shape[0], boxes2.shape[0]))
    for i1 in range(boxes1.shape[0]):
        for i2 in range(boxes2.shape[0]):
            dist[i1, i2] = minDistMinBox(boxes1[i1],
                                         boxes2[i2])
    return dist
