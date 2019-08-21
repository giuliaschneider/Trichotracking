import cv2
import numpy as np
import numpy.linalg
from trichotracking.geometry import (areaAboveLine,
                      areaBelowLine,
                      areaRightOfLine,
                      areaLeftOfLine,
                      getLine,
                      isPointBelowLine)
from trichotracking.regionprops import (calcCentroidMatrix,
                         calcCentroidGlobal,
                         connectContours,
                         getLengths,
                         filterForLargestContour,
                         filterForNLargestContour)
from trichotracking.segmentation import removeNoise

from ._filament import Filament
from ._match_filaments import MatchFilamentEnds

PARAMS_CONTOURS = (cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


class SegmentOverlap():
    """ Segments overlap region and calculates filaments, overlap lengths.

    Keyword arguments:
    img --          gray scale image of cropped filament-filament interaction
    bw --           binary image of cropped filament-filament interaction
    bw_filled --    filled binary image
    background --   background image
    previous --     dictonary of:
                    segmented: segmentated image of previous frame or None
                    overlap_fraction: previous overlap fraction or None
                    cxcy: array centroids of filament ends or None
                    labels: array labels of fil ends or None
                    long_fil: filament object of longer fil or None
                    short_fil: filament object of shorter fil or None
    length --       array of single fil lengths
    connectingLength -- distance up to which single contours are connected
    nbx --          x coord of upper-left corner of cropped img
    nby --          y coord of upper-left corner of cropped img
    getDist --      Function for calculating dist transform
    getInt --       Function for calculating intensity
    seg_functions --Function for segmentation

    """

    def __init__(self, img, bw, bw_filled, background, previous,
                 frame, length, connectingLength, nBx, nBy, getDist,
                 getInt, seg_functions):
        # Initialize variables
        self.img = img
        self.img_shape = img.shape[:2]
        self.bw = bw
        im, self.c_bw, h = cv2.findContours(bw, *PARAMS_CONTOURS)
        self.bw_filled = bw_filled
        self.background = background
        self.previous_segmented = previous["segmented"]
        self.previous_of = previous["overlap_fraction"]
        self.previous_cxcy = previous["cxcy"]
        self.previous_labels = previous["labels"]
        self.previous_long_fil = previous["long_fil"]
        self.previous_short_fil = previous["short_fil"]
        self.frame = frame
        self.length = length
        self.connectingLength = connectingLength
        self.nBx = nBx
        self.nBy = nBy
        self.seg_functions = seg_functions
        self.previous_overlap = self.getPreviousSegmentedOverlap()
        self.previous_dist = self.getPreviousDist()
        self.max_previous_dist = np.max(self.previous_dist)
        self.min_previous_dist = np.min(self.previous_dist)
        self.w_dist = self.getWeightsDensity()
        self.w_int = self.getWeightsIntensity()
        self.dist = getDist(self.bw_filled, self.w_dist)
        self.int = getInt(self.img, self.bw, self.background, self.w_int)
        self.guessOf()

        self.segment()

        self.getLongShortFilaments()
        self.createSegmenedImage()

    def segment(self):
        """ Segments image into regions: background, overlap, fil1, fil2."""
        # More than one region -> # Several filaments, overlap = 0
        if len(self.c_bw) > 1:
            self.segmentSeveralContours()
        else:
            self.chooseSegmentation()

    def segmentSeveralContours(self):
        """ Segmentation if there are several filaments."""
        # Filaments lengths
        bw_fil, c_fil = filterForNLargestContour(2, self.bw, self.c_bw)
        lengths_filaments, _ = getLengths(bw_fil, c_fil)
        indLengths = np.argsort(lengths_filaments)[::-1]
        self.lengths_filaments = lengths_filaments[indLengths]
        self.length_overlap = 0
        self.of = 0
        self.bw_filaments = bw_fil
        self.bw_overlap = np.zeros(self.bw.shape[:2]).astype(np.uint8)
        self.c_filaments = c_fil
        self.labels = np.array([0, 1])
        self.cxcy = calcCentroidGlobal(c_fil, self.nBx, self.nBy)

    def chooseSegmentation(self):
        """ Choses best segmentation. """
        if self.length is None:
            res = self.segmentOneContour(self.seg_functions[0])
        else:
            costs = []
            results = []
            for seg_function in self.seg_functions:
                res = self.segmentOneContour(seg_function)
                costs.append(numpy.linalg.norm(res[0] - self.length))
                results.append(res)
            indBest = np.argmin(np.array(costs))
            res = results[indBest]
            names = [i.__name__ for i in self.seg_functions]

        if costs[indBest] > 40:
            self.segmentNoOverlap(res)
        else:
            self.length_filaments = res[0]
            self.length_overlap = res[1]
            if self.length_filaments[1] != 0:
                self.of = self.length_overlap / self.length_filaments[1]
            else:
                self.of = 0
            self.bw_filaments = res[2]
            self.bw_overlap = res[3]
            self.c_filaments = res[4]
            self.labels = res[5]
            self.cxcy = res[6]

        if not (np.array(self.labels) == 0).any():
            self.labels[:] = [0]

    def segmentOneContour(self, segmentation_function):
        """ Segments the input image in overlap and filament regions. """
        bw_overlap = segmentation_function(self.img_shape, self.dist, self.int)

        # Filters overlap region
        bw_overlap, c_overlap = self.refineOverlap(bw_overlap)

        # Find Filament Endss
        bw_filaments = cv2.subtract(self.bw, bw_overlap)
        bw_filaments = removeNoise(bw_filaments, 15)
        im, c_filaments, h = cv2.findContours(bw_filaments, *PARAMS_CONTOURS)

        # Calculate the total length and centroids of filaments
        self.matching = MatchFilamentEnds(bw_filaments, c_filaments, bw_overlap,
                                          c_overlap, self.previous_cxcy, self.previous_labels,
                                          self.previous_of, self.length, 5, self.nBx, self.nBy)
        length_filaments, length_overlap = self.matching.getLengths()
        labels = self.matching.getLabels()
        cxcy = self.matching.getCentroids()

        return length_filaments, length_overlap, bw_filaments, bw_overlap, \
               c_filaments, labels, cxcy

    def refineOverlap(self, bw_overlap):
        """ Dilates and connects overlap contours."""
        if bw_overlap.any():
            # Dilate
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
            bw_overlap = cv2.dilate(bw_overlap, kernel)
            # Connect separated regions inside connectingLength
            im, c_overlap, h = cv2.findContours(bw_overlap, *PARAMS_CONTOURS)
            c_overlap, bw_overlap = connectContours(bw_overlap, c_overlap,
                                                    self.connectingLength)
            bw_overlap, c_overlap = filterForLargestContour(bw_overlap,
                                                            c_overlap)
            bw_overlap[self.bw == 0] = [0]
            bw_overlap, c_overlap = filterForLargestContour(bw_overlap)
        else:
            c_overlap = []
        return bw_overlap, c_overlap

    def segmentNoOverlap(self, res):
        """ Set variables if overlap can't be determined. """
        self.length_filaments = np.array([np.nan, np.nan])
        self.length_overlap = np.nan
        self.of = 0
        self.bw_filaments = self.bw
        self.bw_overlap = np.zeros(self.img.shape).astype(np.uint8)
        self.c_filaments = np.array([])
        self.labels = np.array([])
        self.cxcy = None

    def getLongShortFilaments(self):
        """ Returns images with the long and short filament. """
        bw_long_fil = self.bw_overlap.copy()
        bw_short_fil = self.bw_overlap.copy()
        for i, c in zip(self.labels, self.c_filaments):
            if i == 0:
                bw_long_fil = cv2.drawContours(bw_long_fil, [c], -1, (255), -1)
            else:
                bw_short_fil = cv2.drawContours(bw_short_fil, [c], -1, (255), -1)
        self.long_fil = Filament(bw_long_fil, self.bw_overlap,
                                 self.previous_long_fil, self.nBx, self.nBy)
        self.short_fil = Filament(bw_short_fil, self.bw_overlap,
                                  self.previous_short_fil, self.nBx, self.nBy)
        self.bw_short_single = self.short_fil.bw_single

    def calcLackOfOverlap(self):
        """ Calculates lack of overlap. """
        # No lack of overlap if overlap fractio = 1
        if self.of == 1 or not self.long_fil.exists:
            xlov, ylov = 0, 0
            bw = np.zeros((self.img.shape[:2])).astype(np.uint8)
        else:
            # Get images of single filaments
            xlov, bw_xlov, A, B, C, D = self.calcXLov()
            ylov, bw_ylov = self.calcYLov(bw_xlov, A, B, C, D)
        return xlov, ylov

    def calcXLov(self):
        """ Calculates the lack of overlap in longitudial direction. """
        # Calculate fraction of shorter filament above / below longer fil
        tl, tr = self.long_fil.tl, self.long_fil.tr
        bl, br = self.long_fil.bl, self.long_fil.br

        if tl[0] - tr[0] == 0:
            if tl[0] > bl[0]:
                upper = areaRightOfLine(self.bw.shape[:2], tl, tr, self.bw_short_single)
                lower = areaLeftOfLine(self.bw.shape[:2], bl, br, self.bw_short_single)
            else:
                upper = areaLeftOfLine(self.bw.shape[:2], tl, tr, self.bw_short_single)
                lower = areaRightOfLine(self.bw.shape[:2], bl, br, self.bw_short_single)
        else:
            upper = areaAboveLine(self.bw.shape[:2], tl, tr, self.bw_short_single)
            lower = areaBelowLine(self.bw.shape[:2], bl, br, self.bw_short_single)

        if upper.any() or lower.any():
            bw_xlov = upper if upper.any() else lower
            length, c_xlov = getLengths(bw_xlov)
            cxcy_lov = calcCentroidMatrix(c_xlov)
            sign = self.long_fil.getDirection(cxcy_lov)
            xlov = sign * length[0]
        else:
            xlov = 0
            bw_xlov = np.zeros((self.img.shape[:2])).astype(np.uint8)
        return xlov, bw_xlov, tl, tr, bl, br

    def calcYLov(self, bw_xlov, A, B, C, D):
        """ Calculates the lack of overlap in lateral direction. """
        # Calculate unaccounted part of short filament
        bw_ylov = cv2.subtract(self.bw_short_single, bw_xlov)
        bw_ylov = removeNoise(bw_ylov, 15)

        # Midline of min area rect of longer filament
        midLine = getLine(0.5 * (A + B), 0.5 * (C + D))

        # Calc lateral lack of overlap
        ylov = 0

        if bw_ylov.any():
            im, c_ylov, h = cv2.findContours(bw_ylov, *PARAMS_CONTOURS)
            cxcy_ylov = calcCentroidMatrix(c_ylov)
            lengths, c_ylov = getLengths(bw_ylov, c_ylov)

            # Iterate trhoug
            for i, c in enumerate(c_ylov):
                isBelow = isPointBelowLine(cxcy_ylov[i, :], *midLine)
                sign = -1 if isBelow else 1
                ylov += sign * lengths[i]
        return ylov, bw_ylov

    def createSegmenedImage(self):
        segmented = np.zeros(self.bw_overlap.shape[:2]).astype(np.uint8)
        segmented[self.bw_overlap == 255] = [1]
        for i, c in zip(self.labels, self.c_filaments):
            col = int((i + 2))
            segmented = cv2.drawContours(segmented, [c], -1, (col), -1)
        self.segmented = segmented

    def calcShortFilPosition(self):
        # Fix !!!!! Make nicer ! #####################################################
        if self.short_fil.exists:
            cxcy_vector = self.short_fil.centroid - self.long_fil.startpoint_gc
            self.short_fil_pos = np.dot(cxcy_vector, self.long_fil.direction)[0]
            self.short_fil_pos = self.short_fil_pos / self.length_filaments[0]
        else:
            self.short_fil_pos = None
        return self.short_fil_pos

    def getPreviousSegmentedOverlap(self):
        """ Returns image of previous overlap or None."""
        if ((self.previous_segmented is not None) and
                (self.previous_segmented == 1).any()):
            previous_overlap = np.zeros(self.img.shape[:2]).astype(np.uint8)
            previous_overlap[self.previous_segmented == 1] = [255]
        else:
            previous_overlap = None
        return previous_overlap

    def getPreviousDist(self):
        """ Returns the distance from outide to the previous overlap."""
        if self.previous_overlap is not None:
            d_prev = cv2.distanceTransform(cv2.bitwise_not(
                self.previous_overlap), cv2.DIST_L2, 3)
        else:
            d_prev = np.ones(self.img.shape[:2]).astype(np.uint8)
        return d_prev

    def getWeightsDensity(self):
        """ Returns the weight matrix for distance thresholding."""
        if self.previous_overlap is not None:
            w_dist = self.previous_dist.copy()
            w_dist[w_dist > 40] = self.max_previous_dist
            w_dist = ((self.max_previous_dist - w_dist) /
                      (self.max_previous_dist - self.min_previous_dist))
        else:
            w_dist = self.previous_dist
        return w_dist

    def getWeightsIntensity(self):
        """ Returns the weight matrix for distance thresholding."""
        if self.previous_overlap is not None:
            w_int = self.previous_dist.copy()
            w_int[w_int > 15] = self.max_previous_dist
            w_int = ((self.max_previous_dist - w_int) /
                     (self.max_previous_dist - self.min_previous_dist)) ** 2
        else:
            w_int = self.previous_dist
        return w_int

    def guessOf(self):
        """ Guess overlapfraction based on dist segmentation. """
        if self.previous_of is None:
            if self.length is not None:
                self.previous_of = 0
                res0 = self.segmentOneContour(self.seg_functions[0])
                self.previous_of = 1
                res1 = self.segmentOneContour(self.seg_functions[0])
                if ((numpy.linalg.norm(res0[0] - self.length)) <
                        (numpy.linalg.norm(res1[0] - self.length))):
                    self.previous_of = 0
            else:
                self.previous_of = 0
                res = self.segmentOneContour(self.seg_functions[0])
                self.length_filaments = res[0]
                self.length_overlap = res[1]
                if self.length_filaments[1] != 0:
                    self.of = self.length_overlap / self.length_filaments[1]
                else:
                    self.of = 0
                self.bw_filaments = res[2]
                self.bw_overlap = res[3]
                self.c_filaments = res[4]
                self.labels = res[5]
                self.cxcy = res[6]
                self.getLongShortFilaments()
                if self.short_fil.exists:
                    diff = np.abs(self.short_fil.orientation - self.long_fil.orientation)
                    if diff < 5:
                        self.previous_of = 1
                else:
                    self.previous_of = 1

    def getAllLengths(self):
        return self.length_filaments, self.length_overlap

    def getSegmentationProps(self):
        seg = dict({"segmented": self.segmented,
                    "overlap_fraction": self.of,
                    "cxcy": self.cxcy,
                    "labels": self.labels,
                    "long_fil": self.long_fil.getFilamentProps(),
                    "short_fil": self.short_fil.getFilamentProps()})
        return seg
