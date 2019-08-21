import numpy as np
import numpy.linalg
from scipy.spatial import distance, distance_matrix


def counterclockwise(A, B, C):
    """ Returns True, if points are ordered counter-clockwise."""
    dca = C - A
    dba = B - A
    return (dca[1]) * (dba[0]) > (dba[1]) * (dca[0])


def isBetween(A, B, C):
    """ Checks if point C is between A and B. """
    d_AB = numpy.linalg.norm(B - A)
    d_AC = numpy.linalg.norm(C - A)
    d_BC = numpy.linalg.norm(C - B)
    if max(d_AC, d_BC) < d_AB:
        return True
    else:
        return False


def intersect(A, B, C, D):
    """ Returns True, if lines AB and CD intersect."""
    return (counterclockwise(A, C, D) != counterclockwise(B, C, D)
            and counterclockwise(A, B, C) != counterclockwise(A, B, D))


def getAngle(A, B, C):
    """ Returns the acute angle between lines AB and AC in radians. """
    # Calculate the angle of individual lines
    y = np.array([B[1] - A[1], C[1] - A[1]])
    x = np.array([B[0] - A[0], C[0] - A[0]])
    a = np.arctan2(y, x) * 180 / np.pi

    # Calculate acute angle
    angle = np.abs(a[0] - a[1])
    if angle > 90:
        angle = 180 - angle
    return angle * np.pi / 180


def getProjection(A, B, C):
    """ Returns the projectioal distance of line AB onto line AC. """
    alpha_BAC = getAngle(A, B, C)
    d_AB = numpy.linalg.norm(B - A)
    return np.cos(alpha_BAC) * d_AB


def sortLongestDistFirst(ordered):
    """ Order such that point with longest distance is at position 0."""
    # Calculates distances between point A and point B/C/D
    nPoints = ordered.shape[0]
    distA = distance_matrix(ordered[0, :][np.newaxis], ordered[1:, :])
    # Calculates distances between point B and point A/C/D
    distB = distance_matrix(ordered[1, :][np.newaxis],
                            np.vstack([ordered[0, :], ordered[2:, :]]))
    # Check for longest distance and adjust ordering of points
    if np.max(distB) >= np.max(distA):
        ordered = np.vstack((ordered[1:, :], ordered[0, :]))
        index = np.append(np.arange(1, nPoints), [0])
    else:
        index = np.arange(0, nPoints)
    return ordered, index


def sortShortestDistFirst(ordered):
    """ Order such that point with longest distance is at position 0."""
    # Calculates distances between point A and point B/C/D
    nPoints = ordered.shape[0]
    distA = distance_matrix(ordered[0, :][np.newaxis], ordered[1:, :])
    # Calculates distances between point B and point A/C/D
    distB = distance_matrix(ordered[1, :][np.newaxis],
                            np.vstack([ordered[0, :], ordered[2:, :]]))
    # Check for longest distance and adjust ordering of points
    if np.min(distB) <= np.min(distA):
        ordered = np.vstack((ordered[1:, :], ordered[0, :]))
        index = np.append(np.arange(1, nPoints), [0])
    else:
        index = np.arange(0, nPoints)
    return ordered, index


def orderCornersRectangle(pts):
    """Returns clockwise ordered points.
       First points is leftmost with longest distance. """

    # Line end points
    lines = [(pts[0, :], pts[1, :], pts[2, :], pts[3, :]), \
             (pts[0, :], pts[2, :], pts[1, :], pts[3, :]), \
             (pts[0, :], pts[3, :], pts[1, :], pts[2, :])]
    indexes = [np.array([0, 1, 2, 3]), np.array([0, 2, 1, 3]), np.array([0, 3, 1, 2])]
    # Find intersecting lines
    intersected = False
    i = -1
    while not intersected and i < len(lines) - 1:
        i += 1
        A, B, C, D = lines[i]
        intersected = intersect(A, B, C, D)
        index = indexes[i]
    if not intersected:
        print("Did not work")

    d_AC = numpy.linalg.norm(C - A)
    d_AD = numpy.linalg.norm(D - A)
    if d_AC > d_AD:
        ordered = np.array([A, C, B, D])
        temp = index[1]
        index[1] = index[2]
        index[2] = temp
    else:
        ordered = np.array([A, D, B, C])
        temp = index[1]
        index[1] = index[3]
        index[3] = index[2]
        index[2] = temp
    # ordered, index2 = sortLongestDistFirst(ordered)
    # index = index[index2]
    return ordered, index


def orderCornersRotatedRectangle(pts):
    """ Return pts as tl, tr, br, bl (func adapted from pyimagesearch)"""
    # Sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # Sort based on y-coordinates
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # point with the largest distance is bottom-right point
    D = distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl])


def orderAllPointsRotatedRectangle(allPts):
    ordered = np.zeros((len(allPts), 4))
    for i, pts in enumerate(allPts):
        ordered[i, :] = orderCornersRotatedRectangle(pts)
    return ordered[:, 0], ordered[:, 1], ordered[:, 2], ordered[:, 3]


def orderCornersTrapezoid(pts):
    """Sort points, A and B are connected by the longest segment. """
    B = pts[1, :].copy()
    D = pts[-1, :].copy()
    d_AB = numpy.linalg.norm(B - pts[0, :])
    d_AD = numpy.linalg.norm(D - pts[0, :])
    if d_AD > d_AB:
        pts[1, :] = D
        pts[-1, :] = B
    return pts


def orderCornersTriangle(pts):
    """Returns clockwise ordered points.
       First points is leftmost with longest distance. """
    ordered, index = sortLongestDistFirst(pts)
    A, B, C = ordered[0, :].copy(), ordered[1, :].copy(), ordered[2, :].copy()
    d_AB = numpy.linalg.norm(B - A)
    d_AC = numpy.linalg.norm(C - A)
    if d_AC > d_AB:
        pts[1, :] = C
        pts[-1, :] = B
    return ordered, index


def getAllDistances(pts):
    """ Returns distances between Points A,B,C,D in array.

    Input: A, B, C, D: center points of filament ends
    Returns:
        AB, AC, BC
        AB, AC, AD, BC, BD, CD
    """
    nPoints = pts.shape[0]
    dist = np.zeros(int(nPoints * (nPoints - 1) / 2))
    adjacencyMatrix = np.zeros((2, int(nPoints * (nPoints - 1) / 2)))
    start = 0
    end = 0
    for i in range(pts.shape[0]):
        nPairs = nPoints - i - 1
        end = start + nPairs
        dist[start:end] = distance_matrix(pts[i, :][np.newaxis], pts[i + 1:, :])
        adjacencyMatrix[0, start:end] = i
        adjacencyMatrix[1, start:end] = np.arange(i + 1, nPoints)
        start = end
    return dist, adjacencyMatrix


def getPairedIndex(pts, pts_overlap=None):
    """ Given three points, return index of points closest & of point furthest."""
    ordered, index = sortShortestDistFirst(pts)
    A, B, C = ordered[0, :].copy(), ordered[1, :].copy(), ordered[2, :].copy()

    if pts_overlap is None:
        d_AB = numpy.linalg.norm(B - A)
        d_AC = numpy.linalg.norm(C - A)

        if d_AB >= d_AC:
            i_furthest = index[1]
            i_pair = np.array([index[0], index[2]])
        else:
            i_furthest = index[2]
            i_pair = np.array([index[0], index[1]])
    else:
        pairs = [(A, B), (A, C), (B, C)]
        i_pairs = [(0, 1), (0, 2), (1, 2)]
        i_furthests = [2, 1, 0]
        angles = np.zeros(3)
        for i, pair in enumerate(pairs):
            angles[i] = getAngle(pts_overlap, pair[0], pair[1])
        i_pair = np.array(i_pairs[i])
        i_furthest = np.array(i_furthests[i])

    return i_pair, i_furthest
