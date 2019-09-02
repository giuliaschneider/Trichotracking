import cv2
import numpy as np

# Index of x, y coordinates in numpy array
NP_YCOORD = 0
NP_XCOORD = 1


def calcArea(contours):
    area = [cv2.contourArea(c) for c in contours]
    area = np.array(area)
    return area


def calcBoundingBox(contours):
    bounding_box = [cv2.boundingRect(c) for c in contours]
    bx = np.array([int(box[0]) for box in bounding_box])
    by = np.array([int(box[1]) for box in bounding_box])
    bw = np.array([int(box[2]) for box in bounding_box])
    bh = np.array([int(box[3]) for box in bounding_box])
    return bounding_box, bx, by, bw, bh


def calcCentroid(moments):
    b = np.array([mom['m00'] for mom in moments])
    ax = np.array([mom['m10'] for mom in moments])
    ay = np.array([mom['m01'] for mom in moments])
    cx = np.divide(ax, b, out=np.zeros_like(ax), where=b != 0)
    cy = np.divide(ay, b, out=np.zeros_like(ay), where=b != 0)
    return cx, cy


def calcCentroidMatrix(contours, moments=None):
    """ Returns matrix of coordiates (r: objects, c:coor in xy coord)."""
    if moments is None:
        moments = [cv2.moments(c) for c in contours]
    cxcy = np.zeros((len(contours), 2))
    cxcy[:, 0], cxcy[:, 1] = calcCentroid(moments)
    return cxcy


def calcCentroidGlobal(c, nBx, nBy):
    """ For a cropped image, returns the centroid in global coord."""
    cxcy = calcCentroidMatrix(c)
    cxcy[:, 0] += nBx
    cxcy[:, 1] += nBy
    return cxcy


def calcConvexArea(convex_hull):
    convex_area = [cv2.contourArea(hull) for hull in convex_hull]
    convex_area = np.array(convex_area)
    return convex_area


def calcEigenvector(contour):
    """ Returns the object center, eigenvectors / -values in xy-coord."""
    size = len(contour)
    data_pts = np.empty((size, 2), dtype=np.float64)
    data_pts[:, 0] = contour[:, 0, 0]
    data_pts[:, 1] = contour[:, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    return mean, eigenvectors, eigenvalues


def calcEigenvalues(contours):
    ew1, ew2 = [], []
    for c in contours:
        ew = calcEigenvector(c)[2]
        if ew.size == 1:
            ew1.append(ew[0][0])
            ew2.append(0)
        else:
            ew1.append(ew[0][0])
            ew2.append(ew[1][0])
    return ew1, ew2


def calcEllipse(contours):
    ellipses = [cv2.fitEllipse(c) if len(c) > 5 else ((0, 0), (0, 0), 0) \
                for c in contours]
    # center = [e[0] for e in ellipses]
    axes = np.array([e[1] for e in ellipses])
    orientation = np.array([e[2] for e in ellipses])
    majoraxis_length = axes[:, 1]
    minoraxis_length = axes[:, 0]
    axRatio = np.divide(minoraxis_length, majoraxis_length,
                        out=np.zeros_like(minoraxis_length),
                        where=majoraxis_length != 0)
    eccentricity = np.sqrt(1 - (axRatio) ** 2)
    return orientation, majoraxis_length, minoraxis_length, eccentricity


def calcMinRect(contours):
    """ Calculates the minimal bounding rectangle for a list of contours.

    Returns:
        min_rect --     center (x,y), (width, height), angle of rotation
        min_box --      four vertices of rect
        min_rect_angle--angle of rotation, angle between horizontal axis
    """

    min_rect = [cv2.minAreaRect(c) for c in contours]
    min_box = [np.int0(cv2.boxPoints(rect)) for rect in min_rect]
    angle = [rect[2] for rect in min_rect]
    min_box_w = [rect[1][0] for rect in min_rect]
    min_box_h = [rect[1][1] for rect in min_rect]
    min_rect_angle = [-(a + 90) if w < h else -a for a, w, h in \
                      zip(angle, min_box_w, min_box_h)]
    return min_rect, min_box, min_rect_angle


def calcLength(min_box):
    """ Calculates the length for a list of contours based on min_rect."""
    # Get box end points
    tl = np.array([box[0] for box in min_box])
    tr = np.array([box[1] for box in min_box])
    br = np.array([box[2] for box in min_box])
    bl = np.array([box[3] for box in min_box])
    # Get box mid points
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    # Get width and heigth, return max value
    dA = np.sqrt((tltrX - blbrX) ** 2 + (tltrY - blbrY) ** 2)
    dB = np.sqrt((tlblX - trbrX) ** 2 + (tlblY - trbrY) ** 2)
    length = np.max([dA, dB], axis=0)
    # print("dA = {}".format(dA))
    # print("dB = {}".format(dB))
    # print("length = {}".format(length))
    return length


def calcPerimeter(contours):
    perimeter = [cv2.arcLength(c, False) for c in contours]
    perimeter = np.array(perimeter)
    return perimeter


def calcPixellist(filledImage, contours, bx, by, bw, bh):
    pixellist_xcoord = []
    pixellist_ycoord = []
    for i in range(len(contours)):
        cfilledImage = filledImage[by[i]:by[i] + bh[i], \
                       bx[i]:bx[i] + bw[i]].copy()
        cfilledImage[cfilledImage != i + 1] = [0]
        pixels = np.nonzero(cfilledImage)
        pixels_ycoord = (pixels[NP_YCOORD] + by[i]).astype(np.int)
        pixels_xcoord = (pixels[NP_XCOORD] + bx[i]).astype(np.int)
        pixellist_xcoord.append(pixels_xcoord.tolist())
        pixellist_ycoord.append(pixels_ycoord.tolist())
    return pixellist_ycoord, pixellist_xcoord


def calcMinDistance(c1, c2):
    """ Calculates the minimum distance between two contours. """
    c1_shape = (c1.shape[0], c1.shape[2])
    c1 = np.array(c1).reshape(c1_shape)
    c2_shape = (c2.shape[0], c2.shape[2])
    c2 = np.array(c2).reshape(c2_shape)
    X1, X2 = np.meshgrid(c1[:, NP_XCOORD], c2[:, NP_XCOORD])
    Y1, Y2 = np.meshgrid(c1[:, NP_YCOORD], c2[:, NP_YCOORD])
    dist = ((X1 - X2) ** 2 + (Y1 - Y2) ** 2) ** 0.5
    return np.min(dist)


def calcSolidity(contours, area, convex_hull, convex_area):
    if convex_area is None:
        if convex_hull is None:
            convex_hull = [cv2.convexHull(c) for c in contours]
        convex_area = calcConvexArea(convex_hull)
    if area is None:
        area = calcArea(contours)
    solidity = np.divide(area, convex_area,
                         out=np.zeros_like(area), where=convex_area != 0)
    return solidity, area, convex_hull, convex_area


def root(id, position):
    while (position != id[position]):
        position = id[position]
    return position


def unite(p, q, clusters):
    pid = clusters[p]
    for i in range(clusters.size):
        if clusters[i] == pid:
            clusters[i] = clusters[q]
    return clusters


def connectContours(img, contours, minLength):
    """ Connects contours based on distance. """
    nContours = len(contours)
    # Assigns each contour to a cluster, save root node in id
    clusters = np.arange(nContours)

    # Iterate through all contour-contour pairs
    for i, c1 in enumerate(contours):
        x = i
        for c2 in contours[i + 1:]:
            x = x + 1
            dist = calcMinDistance(c1, c2)
            if (dist <= minLength):
                unite(x, i, clusters)

    unified = []
    cluster = np.unique(clusters)
    for i in cluster:
        pos = np.where(clusters == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)
    cv2.drawContours(img, unified, -1, 255, -1)
    return unified, img


def drawLine(img, vx, vy, x, y):
    """ Draws a line given by (vx, vy, x, y) into the image. """
    bw = np.zeros(img.shape[:2]).astype(np.uint8)
    cols = img.shape[1]
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(bw, (cols, righty), (0, lefty), (255), 2)
    return bw


def drawRectangle(img, min_box):
    box = np.int0(min_box)
    print(box)
    cv2.drawContours(img, box, -1, (255), 1)
    return img


def getAngleFromMoments(contours):
    """ Angle between horizontal axis and object in degrees. """
    moments = [cv2.moments(c) for c in contours]
    nu11 = np.array([m['nu11'] for m in moments])
    nu20 = np.array([m['nu20'] for m in moments])
    nu02 = np.array([m['nu02'] for m in moments])

    theta = 0.5 * np.arctan2(- 2 * nu11, (nu20 - nu02))
    theta *= 180 / np.pi
    return theta


def getConvexityDefects(img, bw, minLength):
    """ Returns the number of convexity defects longer than a certain length. """
    c = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    areas = calcArea(c)
    indMaxArea = np.argmax(areas)
    contour = c[indMaxArea]
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    dist = np.array([defects[i, 0][3] / 256 for i in range(defects.shape[0])])
    s = np.array([defects[i, 0][0] for i in range(defects.shape[0])])
    e = np.array([defects[i, 0][1] for i in range(defects.shape[0])])
    s_e = np.abs(s - e)
    dist = dist[(dist > minLength) & (s_e > minLength)]
    return dist.size


def getExtremes(contours):
    """ Returns a list of extremes in xy-coordinates."""
    leftmost = [tuple(c[c[:, :, 0].argmin()][0]) for c in contours]
    rightmost = [tuple(c[c[:, :, 0].argmax()][0]) for c in contours]
    topmost = [tuple(c[c[:, :, 1].argmin()][0]) for c in contours]
    bottomost = [tuple(c[c[:, :, 1].argmax()][0]) for c in contours]
    return leftmost, rightmost, topmost, bottomost


def getFourPeriodicContourNeighborIndexes(c, i, R):
    nPoints = len(c)
    neighborIndexes = []
    for offset in range(1, R):
        if i - offset < 0:
            neighborIndexes.append(nPoints + (i - offset))
        else:
            neighborIndexes.append(i - offset)
        if i + offset > nPoints - 1:
            neighborIndexes.append((i + offset) - nPoints)
        else:
            neighborIndexes.append(i + offset)

    return neighborIndexes


def getLength(bw, c=None):
    """ Returns the length of the largest contour in bw."""
    if c is None:
        c, bw = filterForLargestContour(bw)
        c = [c]
    min_rect, min_box, _ = calcMinRect(c)
    min_rect = min_rect[0]
    w = min_rect[1][0]
    h = min_rect[1][1]
    return max(w, h), c, bw, min_rect, min_box


def getLengths(bw, c=None):
    if c is None:
        c = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    # length = calcPerimeter(c)/2
    if len(c) > 0:
        min_box = calcMinRect(c)[1]
        length = calcLength(min_box)
    else:
        length = np.array([0])
    return length, c


def filterForLargestContour(bw, c=None):
    """ Returns bw image with just the largest contour."""
    if c is None:
        c = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    areas = calcArea(c)
    indMaxArea = np.argmax(areas)
    bwFiltered = np.zeros(bw.shape[0:2]).astype(np.uint8)
    cv2.drawContours(bwFiltered, c, indMaxArea, (255), -1)
    return bwFiltered, [c[indMaxArea]]


def filterForNLargestContour(n, bw, c=None):
    """ Returns bw image with just the largest contour."""
    if c is None:
        c = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    areas = calcArea(c)
    indMaxAreas = np.argsort(areas)[::1][:n]
    bwFiltered = np.zeros(bw.shape[0:2]).astype(np.uint8)
    for i in indMaxAreas:
        cv2.drawContours(bwFiltered, c, i, (255), -1)
    cFiltered = [c[i] for i in indMaxAreas]
    return bwFiltered, cFiltered


def insideROI(min_box, roi):
    """ Returns bool array, if each row is inside roi. """
    n = min_box.shape[0]
    inRoi = []
    for i in range(n):
        if (((min_box[i][:, 1] < roi.shape[0]).all()) and
                ((min_box[i][:, 0] < roi.shape[1]).all())):
            inRoi.append(roi[min_box[i][:, 1] - 1, min_box[i][:, 0] - 1].all())
        else:
            inRoi.append(False)
    return np.array(inRoi).astype(np.bool)


def midpoint(ptA, ptB):
    mps = 0.5 * (ptA + ptB)
    return mps[:, 0], mps[:, 1]
