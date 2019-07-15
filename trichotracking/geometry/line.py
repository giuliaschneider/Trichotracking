import numpy as np

def getLine(A, B):
    """ Returns parameters for line between points A and B.  """
    x0, y0 = A[0], A[1]
    x1, y1 = B[0], B[1]
    vy = y1 - y0
    vx = x1 - x0
    return vx, vy, x0, y0

def isBelowLine(img, vx, vy, x0, y0):
    """ Returns bool matrix which states if an element is above the line."""
    rows, cols = img.shape[:2]
    x = np.arange(0,cols)
    y = np.arange(0, rows)
    X, Y = np.meshgrid(x,y)
    return Y > vy/vx*(X-x0) +  y0

def isRigthOfLine(img, vx, vy, x0, y0):
    rows, cols = img.shape[:2]
    x = np.arange(0,cols)
    y = np.arange(0, rows)
    X, Y = np.meshgrid(x,y)
    return X > vx/vy*(Y-y0) +  x0

def isPointBelowLine(pt, vx, vy, x0, y0):
    """ Returns bool which states if point is above the line."""
    x, y = pt[0], pt[1]
    return y > vy/vx*(x-x0) +  y0

def areaAboveLine(img_shape, A, B, mask=None):
    """ Returns binary image with all points above line A-B true."""
    bw = np.zeros((img_shape)).astype(np.uint8)
    line = getLine(A, B)
    if mask is not None:
        condition = (~(isBelowLine(bw, *line)) & (mask==255))
    else:
        condition = ~(isBelowLine(bw, *line))
    bw[condition] = [255]
    return bw

def areaBelowLine(img_shape, A, B, mask=None):
    """ Returns binary image with all points above line A-B true."""
    bw = np.zeros((img_shape)).astype(np.uint8)
    line = getLine(A, B)
    if mask is not None:
        condition = (isBelowLine(bw, *line) & (mask==255))
    else:
        condition = isBelowLine(bw, *line)
    bw[condition] = [255]
    return bw

def areaRightOfLine(img_shape, A, B, mask=None):
    """ Returns binary image with all points above line A-B true."""
    bw = np.zeros((img_shape)).astype(np.uint8)
    line = getLine(A, B)
    if mask is not None:
        condition = ((isRigthOfLine(bw, *line)) & (mask==255))
    else:
        condition = (isRigthOfLine(bw, *line))
    bw[condition] = [255]
    return bw

def areaLeftOfLine(img_shape, A, B, mask=None):
    """ Returns binary image with all points above line A-B true."""
    bw = np.zeros((img_shape)).astype(np.uint8)
    line = getLine(A, B)
    if mask is not None:
        condition = (~(isRigthOfLine(bw, *line)) & (mask==255))
    else:
        condition = ~(isRigthOfLine(bw, *line))
    bw[condition] = [255]
    return bw
