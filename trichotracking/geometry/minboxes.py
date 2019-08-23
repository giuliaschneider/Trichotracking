import numpy as np
import numpy.linalg as la

from .points import getAngle, intersect


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


# Checks if point faces edge
def point_faces_edge(edge, point):
    vEdge = edge[1] - edge[0]
    vPoint = point - edge[0]
    c = np.dot(vPoint, vEdge) / (la.norm(vEdge) * la.norm(vPoint))
    theta1 = np.arccos(np.clip(c, -1, 1))
    vEdge = edge[0] - edge[1]
    vPoint = point - edge[1]
    c = np.dot(vPoint, vEdge) / (la.norm(vEdge) * la.norm(vPoint))
    theta2 = np.arccos(np.clip(c, -1, 1))

    return theta1 <= np.pi / 2 and theta2 <= np.pi / 2


def edgePoints(box):
    return [[box[0, :], box[1, :]], [box[1, :], box[2, :]], [box[2, :], box[3, :]], [box[3, :], box[0, :]]]


def distance_between_edge_and_point(edge, point):  # edge is a tupple of points
    if point_faces_edge(edge, point):
        vEdge = edge[1] - edge[0]
        dEdge = la.norm(edge[1] - edge[0])
        return (vEdge[1] * point[0] - vEdge[0] * point[1] + edge[1][0] * edge[0][1] - edge[1][1] * edge[0][0]) / dEdge
    return min(la.norm(edge[0] - point), la.norm(edge[1] - point))


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

    SOURCE: https://github.com/Pithikos/python-rectangles/blob/master/geometry.py
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

    # 2. draw a line between rectangles
    line = (c1, c2)

    # 3. find the two edges that intersect the line
    edge1 = None
    edge2 = None
    for edge in edgePoints(box1):
        if intersect(*edge, *line):
            edge1 = edge
            break
    for edge in edgePoints(box2):
        if intersect(*edge, *line):
            edge2 = edge
            break

    # 4. find shortest distance between these two edges
    distances = [
        distance_between_edge_and_point(edge1, edge2[0]),
        distance_between_edge_and_point(edge1, edge2[1]),
        distance_between_edge_and_point(edge2, edge1[0]),
        distance_between_edge_and_point(edge2, edge1[1]),
    ]

    return min(distances)


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
