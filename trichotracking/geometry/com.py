def calcCenterOfMass(r1, a1, r2, a2):
    cxcy = (r1 * a1 + r2 * a2) / (a1 + a2)
    cx, cy = cxcy[:, 0], cxcy[:, 1]
    return cx, cy
