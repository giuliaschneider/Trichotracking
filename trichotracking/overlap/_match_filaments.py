import numpy as np
import numpy.linalg

from trichotracking.geometry import (getPairedIndex,
                                     orderCornersRectangle)
from trichotracking.linking import matcher
from trichotracking.regionprops import (calcCentroidGlobal,
                                        filterForNLargestContour,
                                        getAngleFromMoments,
                                        getLengths)


class MatchFilamentEnds():
    """ Assigns filament ends to filaments and calculates length of filaments.

    Keyword arguments:
    bw_singles -- binary image of single filaments
    c_singles --  contours of filaments
    bw_overlap --   binary image of overlap
    c_overlap --    contours of overlap
    cxcy_previous --centroids of filaments of previous time step
    labels_prev --  filament labels of previous time step
    maxDist --      maximal matching distance
    nBx --          x-Coordinate of top-left corner of cropped image
    nBy --          y-Coordinte of top-left corner of  cropped image
    """

    def __init__(self, bw_singles, c_singles, bw_overlap, c_overlap,
                 cxcy_previous, labels_prev, of_previous, avg, maxDist,
                 nBx, nBy):
        self.bw_singles = bw_singles
        self.bw_singles, self.c_singles = self.getFilSingles(c_singles)
        self.bw_overlap = bw_overlap
        self.c_overlap = c_overlap
        self.cxcy_previous = cxcy_previous
        self.labels_prev = labels_prev
        self.of_previous = of_previous
        self.avg = avg
        self.maxDist = maxDist
        self.nBx = nBx
        self.nBy = nBy

        self.cxcy_current = calcCentroidGlobal(self.c_singles, nBx, nBy)
        self.cxcy_overlap = calcCentroidGlobal(self.c_overlap, nBx, nBy)
        if self.cxcy_overlap.shape[0] >= 1:
            self.cxcy_overlap = self.cxcy_overlap[0]

        self.length_singles, self.length_ov = self.calcLengths()
        self.nSingles = len(self.length_singles)

        self.matchAll()

    def matchAll(self):
        if len(self.cxcy_overlap) < 1:
            self.matchNoOverlap()
        elif self.cxcy_previous is None:
            self.matchFirstFrame()
        else:
            self.labels_curr = np.zeros((self.nSingles)) - 1
            indMP, indMC = matcher(self.cxcy_previous[:, 0],
                                   self.cxcy_previous[:, 1],
                                   self.cxcy_current[:, 0],
                                   self.cxcy_current[:, 1], self.maxDist)
            self.labels_curr[indMC] = self.labels_prev[indMP]
            indNMC = np.where(self.labels_curr < 0)[0]
            if indNMC.size > 0:
                self.matchNotMatched(indMC, indNMC)

    def matchNoOverlap(self):
        self.labels_curr = np.zeros(1)

    def matchFirstFrame(self):
        """ Matches filaments ends based on geometric relations."""
        if (self.nSingles == 1):
            self.matchOneFilamentEnd()
        elif (self.nSingles == 2):
            self.matchTwoFilamentEnds()
        elif (self.nSingles == 3):
            self.matchThreeFilamentEnds()
        elif (self.nSingles == 4):
            self.matchFourFilamentEnds()

    def matchNotMatched(self, indMC, indNMC):
        if indNMC.size == 1:
            if (self.nSingles == 1):
                self.matchOneFilamentEnd()
            elif (self.nSingles == 3):
                self.matchOneThreeFilamentEnds(indMC, indNMC)
            else:
                self.matchOneManyFilamentEnds(indMC, indNMC)
        elif indNMC.size == 2:
            if (self.nSingles == 2):
                self.matchTwoTwoFilamentEnds(indMC, indNMC)
            elif (self.nSingles == 3):
                self.matchTwoThreeFilamentEnds(indMC, indNMC)
            else:
                self.matchTwoFourFilamentEnds(indMC, indNMC)
        elif indNMC.size == 3:
            if (self.nSingles == 3):
                self.matchThreeThreeFilamentEnds()
            else:
                self.matchThreeFourFilamentEnds(indMC, indNMC)
        else:
            self.matchFourFilamentEnds()

    def matchOneFilamentEnd(self):
        self.labels_curr = np.array([0])

    def matchOneManyFilamentEnds(self, indMC, indNMC):
        lengths = self.getMatchedLengths(indMC, indNMC)
        l = self.length_singles[indNMC][0]
        comb1 = lengths + np.array([l, 0])
        comb2 = lengths + np.array([0, l])
        comb = self.matchBetweenTwoLength(self.avg, comb1, comb2)
        if comb == 0:
            self.labels_curr[indNMC] = 0
        else:
            self.labels_curr[indNMC] = 1

    def matchOneThreeFilamentEnds(self, indMC, indNMC):
        i_pair, i_furthest = getPairedIndex(self.cxcy_current,
                                            self.cxcy_overlap)
        if indNMC[0] == i_furthest:
            self.matchOneManyFilamentEnds(indMC, indNMC)
        else:
            indM = int(np.where(indMC != i_furthest)[0])
            matched_pair = indMC[indM]
            indNM = int(np.where(i_pair != matched_pair)[0])
            not_matched = i_pair[indNM]
            self.labels_curr[not_matched] = \
                int(not self.labels_curr[matched_pair])

    def matchTwoFilamentEnds(self):
        self.labels_curr = np.array([0, 0])
        if self.avg is not None:
            l_singles = self.length_singles
            lengths = np.array([
                l_singles[0] + self.length_ov + l_singles[1],
                self.length_ov])
            lengths2 = np.array([
                l_singles[0] + self.length_ov,
                self.length_ov + l_singles[1]])
            I = np.argsort(lengths2)[::-1]
            lengths2 = lengths2[I]
            ind2 = np.array([0, 1])[I]
            comb = self.matchBetweenTwoLength(self.avg, lengths, lengths2)
            if comb == 1:
                self.labels_curr[ind2] = np.array([0, 1])


        elif self.of_previous < 0.8:
            index_fil = np.argsort(self.length_singles)[::-1]
            self.labels_curr[index_fil[1]] = 1

    def matchTwoTwoFilamentEnds(self, indMC, indNMC):
        indMP, indMC = matcher(self.cxcy_previous[:, 0],
                               self.cxcy_previous[:, 1],
                               self.cxcy_current[:, 0],
                               self.cxcy_current[:, 1], self.maxDist * 25)
        if len(indMP) > 0:
            indMP = np.array([indMP[0]])
            indMC = np.array([indMC[0]])
            self.labels_curr[indMC] = self.labels_prev[indMP]
            indNMC = np.where(self.labels_curr < 0)[0]
            self.matchOneManyFilamentEnds(indMC, indNMC)
        else:
            self.matchTwoFilamentEnds()

    def matchTwoThreeFilamentEnds(self, indMC, indNMC):
        lengths = self.getMatchedLengths(indMC, indNMC)
        i_pair, i_furthest = getPairedIndex(self.cxcy_current,
                                            self.cxcy_overlap)
        lengthsNM = np.array([self.length_singles[indNMC[0]],
                              self.length_singles[indNMC[1]]])
        comb1 = lengths + lengthsNM
        comb2 = lengths + lengthsNM[::-1]
        comb3 = lengths
        comb3[int(not self.labels_curr[indMC])] += np.sum(lengthsNM)

        if i_furthest in indNMC:
            indM = indNMC[np.where(indNMC != i_furthest)[0]]
            self.labels_curr[indM] = int(not (self.labels_curr[indMC]))
            indMC = np.append(indMC, indM)
            self.matchOneThreeFilamentEnds(indMC, np.array([i_furthest]))
        else:
            comb = self.matchBetweenTwoLength(self.avg, comb1, comb2)
            if comb == 0:
                self.labels_curr[indNMC] = [0, 1]
            elif comb == 1:
                self.labels_curr[indNMC] = [1, 0]
            else:
                label = int(not self.labels_curr[indMC])
                self.labels_curr[indNMC] = [label, label]

    def matchTwoFourFilamentEnds(self, indMC, indNMC):
        pts, ind = orderCornersRectangle(self.cxcy_current)
        lengths = self.getMatchedLengths(indMC, indNMC)
        if (lengths == self.length_ov).any():
            ind = np.where(lengths == self.length_ov)[0]
            self.labels_curr[indNMC] = int(not self.labels_curr[indMC[0]])
        else:
            lengthsNM = np.array([self.length_singles[indNMC[0]],
                                  self.length_singles[indNMC[1]]])
            comb1 = lengths + lengthsNM
            comb2 = lengths + lengthsNM[::-1]
            comb = self.matchBetweenTwoLength(self.avg, comb1, comb2)
            if comb == 0:
                self.labels_curr[indNMC] = [0, 1]
            else:
                self.labels_curr[indNMC] = [1, 0]

    def matchThreeFilamentEnds(self):
        i_pair, i_furthest = getPairedIndex(self.cxcy_current, self.cxcy_overlap)
        self.labels_curr = np.array([0, 0, 0])

        if self.avg is not None:
            l_singles = self.length_singles
            lengths = np.array([
                l_singles[i_furthest] + self.length_ov + l_singles[i_pair[0]],
                self.length_ov + l_singles[i_pair[1]]])
            indP = np.array([0, 1])
            I = np.argsort(lengths)[::-1]
            lengths = lengths[I]
            indP = indP[I]
            indF = I[0]
            lengths2 = np.array([
                l_singles[i_furthest] + self.length_ov + l_singles[i_pair[1]],
                self.length_ov + l_singles[i_pair[0]]])
            indP2 = np.array([1, 0])
            I = np.argsort(lengths2)[::-1]
            lengths2 = lengths2[I]
            indP2 = indP2[I]
            indF2 = I[0]
            comb = self.matchBetweenTwoLength(self.avg, lengths, lengths2)
            if comb == 1:
                lengths = lengths2
                indP = indP2
                indF = indF2
            self.labels_curr[i_pair] = indP
            self.labels_curr[i_furthest] = indF
        else:
            angles = getAngleFromMoments(self.c_singles)
            da_AB = np.abs(angles[i_furthest] - angles[i_pair[0]])
            da_AC = np.abs(angles[i_furthest] - angles[i_pair[1]])
            if da_AB > da_AC:
                self.labels_curr[i_pair[0]] = 1
            else:
                self.labels_curr[i_pair[1]] = 1

            # Check if single filament end belongs to long fil
            indMC = [i for i in range(0, self.labels_curr.size)]
            indNMC = []
            lengths = self.getMatchedLengths(indMC, indNMC)
            if lengths[0] < lengths[1]:
                self.switchLabels()

    def matchThreeThreeFilamentEnds(self):
        indMP, indMC = matcher(self.cxcy_previous[:, 0],
                               self.cxcy_previous[:, 1],
                               self.cxcy_current[:, 0],
                               self.cxcy_current[:, 1], self.maxDist * 25)
        if len(indMP) > 0:
            indMP = np.array([indMP[0]])
            indMC = np.array([indMC[0]])
            self.labels_curr[indMC] = self.labels_prev[indMP]
            indNMC = np.where(self.labels_curr < 0)[0]
            self.matchTwoThreeFilamentEnds(indMC, indNMC)
        else:
            self.matchThreeFilamentEnds()

    def matchThreeFourFilamentEnds(self, indMC, indNMC):

        pts, ind = orderCornersRectangle(self.cxcy_current)
        lengths = self.getMatchedLengths(indMC, indNMC)

        i_pair1 = np.array([ind[0], ind[3]])
        i_pair2 = np.array([ind[1], ind[2]])

        if indMC in i_pair1:
            ind = i_pair1[np.where(i_pair1 != indMC)]
        else:
            ind = i_pair2[np.where(i_pair2 != indMC)]
        self.labels_curr[ind] = int(not self.labels_curr[indMC])
        indMC = np.append(indMC, ind)
        indNMC = indNMC[indNMC != ind]
        self.matchTwoFourFilamentEnds(indMC, indNMC)

    def matchFourFilamentEnds(self):
        pts, ind = orderCornersRectangle(self.cxcy_current)
        l_singles = self.length_singles
        self.labels_curr = np.zeros((4))
        lengths = np.array(
            [l_singles[ind[0]] + l_singles[ind[1]] + self.length_ov,
             l_singles[ind[2]] + l_singles[ind[3]] + self.length_ov])
        indl = np.array([0, 1, 2, 3])
        if self.avg is not None:
            lengths2 = np.array(
                [l_singles[ind[0]] + l_singles[ind[2]] + self.length_ov,
                 l_singles[ind[1]] + l_singles[ind[3]] + self.length_ov])
            indl2 = np.array([0, 2, 1, 3])
            comb = self.matchBetweenTwoLength(self.avg, lengths, lengths2)
            if comb == 1:
                lengths = lengths2
                indl = indl2
        if lengths[0] > lengths[1]:
            self.labels_curr[np.r_[ind[indl[2]], ind[indl[3]]]] = 1
        else:
            self.labels_curr[np.r_[ind[indl[0]], ind[indl[1]]]] = 1

    def matchBetweenTwoLength(self, avg, comb1, comb2):
        diff1 = numpy.linalg.norm(comb1 - avg)
        diff2 = numpy.linalg.norm(comb2 - avg)
        if diff1 < diff2:
            comb = 0
        else:
            comb = 1
        return comb

    def matchBetweenThreeLength(self, avg, comb1, comb2, comb3):
        diff1 = np.sum(np.abs(comb1 - avg))
        diff2 = np.sum(np.abs(comb2 - avg))
        diff3 = np.sum(np.abs(comb3 - avg))
        return np.argmin(np.array([diff1, diff2, diff3]))

    def calcLengths(self):
        length_singles, _ = getLengths(self.bw_singles, self.c_singles)
        length_singles = np.array(length_singles)
        length_ov, _ = getLengths(self.bw_overlap, self.c_overlap)
        length_ov = length_ov[0]
        return length_singles, length_ov

    def getFilSingles(self, c_singles):
        """ Checks if there are more than 4 fil ends. """
        if len(c_singles) > 4:
            bw_singles, c_singles = filterForNLargestContour(
                4, self.bw_singles, c_singles)
        else:
            bw_singles = self.bw_singles
        return bw_singles, c_singles

    def getLengths(self):
        length_filaments = np.zeros(2) + self.length_ov
        for i in range(self.nSingles):
            ind = int(self.labels_curr[i])
            length_filaments[ind] += self.length_singles[i]
        return length_filaments, self.length_ov

    def getMatchedLengths(self, indMC, indNMC):
        lengths = np.zeros(2) + self.length_ov
        for i in indMC:
            ind = int(self.labels_curr[i])
            lengths[ind] += self.length_singles[i]
        return lengths

    def switchLabels(self):
        for i in range(self.labels_curr.size):
            self.labels_curr[i] = int(not self.labels_curr[i])

    def getLabels(self):
        return self.labels_curr

    def getCentroids(self):
        return self.cxcy_current
