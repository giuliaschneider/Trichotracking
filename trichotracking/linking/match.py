import numpy as np


def getDistMatrix(cxPrevious, cyPrevious, cxCurrent, cyCurrent):
    """Returns dist matrix of all particle in previous and current time."""
    xPrev, xCurr = np.meshgrid(cxPrevious, cxCurrent)
    yPrev, yCurr = np.meshgrid(cyPrevious, cyCurrent)
    dist = (xCurr - xPrev) ** 2 + (yCurr - yPrev) ** 2
    return dist


def matcher(cxPrevious, cyPrevious, cxCurrent, cyCurrent, maxDist,
            lengthPrevious=None, lengthCurrent=None, maxdLength=None,
            indP1=None, indP2=None, indC=None,
            areaPrevious=None, areaCurrent=None, maxdArea=None):
    """ Matches based on nearest distance, returns indexes of match.

    Positional arguments:
    cxPrevious --   x coord. of particles of previous time, np.array
    cyPrevious --   y coord. of particles of previous time, np.array
    cxCurrent --    x coord. of particles of current time, np.array
    cyCurrent --    y coord. of particles of current time, np.array
    maxDist --      Maximal searching distance

    Keyword arguments:
    lengthPrevious--Length of particles of previous time, np.array
    lengthCurrent --Length of particles of current time, np.array
    maxdLength --   Maximal length difference

    areaPrevious -- Area of particles of previous time, np.array
    areaCurrent -- Area of particles of current time, np.array
    maxdArea --   Maximal area difference

    indP1  --       TrackNrs previous 1
    indP2 --        trackNrs previous 2
    indC --         trackNrs current


    Returns:
    indPrevious -- list of indexes of matched particles previous time
    indCurrent  -- list of indexes of matched particles current time
    """

    # Calculate distance between every particle
    # from previous and current time step
    dist = getDistMatrix(cxPrevious, cyPrevious, cxCurrent, cyCurrent)

    # Additional length conditionmatcher
    if lengthPrevious is not None:
        lPrev, lCurr = np.meshgrid(lengthPrevious, lengthCurrent)
        dl = np.abs(lCurr - lPrev)
        dist[dl > maxdLength] = dist.max()

    # Additional area conditon
    if areaPrevious is not None:
        aPrev, aCurr = np.meshgrid(areaPrevious, areaCurrent)
        da = np.abs(aCurr - aPrev)
        dist[da > maxdArea] = dist.max()

    # Get index of previous and current particle
    # which are closer than maxDist
    rowIndexCurr, colIndexPrev = np.where(dist < maxDist ** 2)
    v = dist[rowIndexCurr, colIndexPrev]

    # Sort index according to closest distance
    vSortedIndex = np.argsort(v)
    indCurrent = rowIndexCurr[vSortedIndex]
    indPrevious = colIndexPrev[vSortedIndex]

    # Get boolean array masking duplicate indexes
    mCurrent = np.zeros_like(indCurrent, dtype=bool)
    mCurrent[np.unique(indCurrent, return_index=True)[1]] = True
    mPrevious = np.zeros_like(indPrevious, dtype=bool)
    mPrevious[np.unique(indPrevious, return_index=True)[1]] = True
    maskDuplicates = mCurrent & mPrevious

    # Check if any duplicates in trackN
    if indP1 is not None:
        indP1 = indP1[indPrevious]
        mP1 = np.zeros_like(indP1, dtype=bool)
        mP1[np.unique(indP1, return_index=True)[1]] = True
        maskDuplicates = maskDuplicates & mP1
    if indP2 is not None:
        indP2 = indP2[indPrevious]
        mP2 = np.zeros_like(indP2, dtype=bool)
        mP2[np.unique(indP2, return_index=True)[1]] = True
        maskDuplicates = maskDuplicates & mP2
    if indC is not None:
        indC = indC[indCurrent]
        mC = np.zeros_like(indC, dtype=bool)
        mC[np.unique(indC, return_index=True)[1]] = True
        maskDuplicates = maskDuplicates & mC

    # delete duplicates
    # indexes of previous&current matched particles, not sorted
    indCurrent = indCurrent[maskDuplicates]
    indPrevious = indPrevious[maskDuplicates]

    # print("Matched {} of {} particles".format(len(indPrevious), len(cxPrevious)))
    return list(indPrevious), list(indCurrent)
