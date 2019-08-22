import cv2
import numpy as np
from numpy.linalg import norm

from trichotracking.linking import matcher
from trichotracking.regionprops import (calcCentroidGlobal,
                                        calcEigenvector,
                                        calcMinRect,
                                        getAngleFromMoments)

PARAMS_CONTOURS = (cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


class Filament():
    """ Stores filament information.

    Keyword arguments:
    bw --           binary image of single filament
    """

    def __init__(self, bw, bw_overlap, previous_fil, nBx, nBy):
        # Initialize variables
        self.bw = bw
        self.bw_overlap = bw_overlap
        self.bw_single = cv2.subtract(bw, bw_overlap)
        self.initializePreviousVariables(previous_fil)
        self.nBx = nBx
        self.nBy = nBy
        self.size = max(self.bw.shape)
        self.exists = self.bw.any()

        if self.exists:
            im, self.contour, h = cv2.findContours(self.bw, *PARAMS_CONTOURS)
            self.getAxisLongFilament()
            self.setMinRect()
            self.findStartpoint()
            self.direction = self.mean - self.startpoint_lc
            self.direction = self.direction / norm(self.direction)
            self.setOrientation()
            self.setCentroid()

    def initializePreviousVariables(self, previous_fil):
        if previous_fil is None:
            self.previous_orientation = None
            self.previous_startpoint = None
            self.previous_centroid = None
        else:
            self.previous_orientation = previous_fil["orientation"]
            self.previous_startpoint = previous_fil["startpoint"]
            self.previous_centroid = previous_fil["centroid"]

    def getAxisLongFilament(self):
        # Calculate eigenvector of shape
        mean, eigenvectors, eigenvalues = calcEigenvector(self.contour[0])
        self.mean = mean[0]
        self.eigenvector1 = eigenvectors[0, :]
        self.eigenvector2 = eigenvectors[1, :]

        # Line properties of axis
        self.axis = (self.eigenvector1[0], self.eigenvector1[1],
                     self.mean[0], self.mean[1])

    def setMinRect(self):
        # Min area rect of long filament
        # min_box points: 0 min_box[0]:lowest point, rest clockwise
        # height is distance between poiself,self.long_fil, self.short_fil, nts 0 and 1

        self.min_rect, min_box, min_rect_angle = calcMinRect(self.contour)
        self.min_rect_w, self.min_rect_h = self.min_rect[0][1]
        self.min_box = min_box[0]
        self.min_rect_angle = min_rect_angle[0]

        if self.min_rect_h < self.min_rect_w:
            self.tl, self.tr = self.min_box[2, :], self.min_box[3, :]  # Top points
            self.bl, self.br = self.min_box[0, :], self.min_box[1, :]  # Bottom points
        else:
            self.tl, self.tr = self.min_box[1, :], self.min_box[2, :]  # Top points
            self.bl, self.br = self.min_box[0, :], self.min_box[3, :]  # Bottom points
        self.cxcy_endpoints = np.vstack((0.5 * (self.tl + self.tr),
                                         0.5 * (self.bl + self.br)))
        self.cxcy_endpoints[:, 0] += self.nBx
        self.cxcy_endpoints[:, 1] += self.nBy

    def findStartpoint(self):
        if self.previous_startpoint is None:
            self.startpoint_gc = self.cxcy_endpoints[0, :]
        else:
            indMP, indMC = matcher(self.previous_startpoint[0],
                                   self.previous_startpoint[1],
                                   self.cxcy_endpoints[:, 0],
                                   self.cxcy_endpoints[:, 1], self.size * 3)
            if len(indMP) <= 0:
                set_trace()
            self.startpoint_gc = self.cxcy_endpoints[indMC, :][0]
        self.startpoint_lc = self.startpoint_gc.copy()
        self.startpoint_lc[0] -= self.nBx
        self.startpoint_lc[1] -= self.nBy
        # print("Startpoint lc = {}".format(self.startpoint_lc))
        # set_trace()

    def setOrientation(self):
        at1 = getAngleFromMoments(self.contour)[0]
        at0 = self.previous_orientation
        if ((at0 is not None) and (abs(at1 - at0) > abs(abs(at1) - abs(at0)))):
            sign = np.sign(at0)
            at1 = at1 + sign * 180
        self.orientation = at1

    def setCentroid(self):
        self.centroid = calcCentroidGlobal(self.contour, self.nBx, self.nBy)

    def getDirection(self, pt):
        # print("Direction = {}".format(self.direction))
        # print(pt)
        # print((pt- self.startpoint_lc))
        # print(np.dot( (pt- self.startpoint_lc), self.direction ))
        return np.sign(np.dot((pt - self.mean), self.direction))[0]

    def getFilamentProps(self):
        if self.exists:
            fil = dict({"orientation": self.orientation,
                        "startpoint": self.startpoint_gc,
                        "centroid": self.centroid,
                        "direction": self.direction})
        else:
            fil = None
        return fil
