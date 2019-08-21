import multiprocessing as mp

import pandas as pd
from trichotracking.trackkeeper import Trackkeeper
from trichotracking.utility import split_list

from .match import matcher


def link(dfTracks,
         maxLinkTime=3,
         maxLinkDist=10):
    print("Start Linking")

    processes = mp.cpu_count()
    frames = split_list(dfTracks.frame.unique().tolist(), processes)
    dfs = [dfTracks[dfTracks.frame.isin(frames[i])] for i in range(processes)]

    pool = mp.Pool(processes=processes)
    results = [pool.apply_async(link_part,
                                args=(dfs[x],maxLinkTime, maxLinkDist)) for x in range(processes)]

    pool.close()
    pool.join()

    dfs = [result.get() for result in results]
    dfTracks = pd.concat(dfs, ignore_index=True)
    dfTracks = link_part(dfTracks)
    return dfTracks


def link_part(dfTracks,
              maxLinkTime=3,
              maxLinkDist=10,
              maxdLength=15,
              maxdArea=80):
    """ Links track segments by connecting ends to starts. """

    keeper = Trackkeeper.fromDf(dfTracks, None, None)
    print("Link dist = {}".format(maxLinkDist))

    # Iterate through linking time steps
    for i in range(1, maxLinkTime + 1):
        endTimes = keeper.getEndTimes()

        # Iterate through unique endtimes
        for endTime in endTimes:
            startTime = endTime + i
            startTracks = keeper.getStartTracks(startTime)
            endTracks = keeper.getEndTracks(endTime)

            # Link based on distance and difference in length
            if (endTracks.size > 0) and (startTracks.size > 0):
                dfP = keeper.getTracksAtTime(endTime, endTracks)
                trackNrPrev = dfP.trackNr.values
                cxPrev = dfP.cx.values
                cyPrev = dfP.cy.values
                lPrev = dfP.length.values
                aPrev = dfP.area.values

                dfC = keeper.getTracksAtTime(startTime, startTracks)
                trackNrCurr = dfC.trackNr.values
                cxCurr = dfC.cx.values
                cyCurr = dfC.cy.values
                lCurr = dfC.length.values
                aCurr = dfC.area.values

                indMP, indMC = matcher(cxPrev,
                                       cyPrev,
                                       cxCurr,
                                       cyCurr,
                                       maxLinkDist,
                                       # lengthPrevious=lPrev,
                                       # lengthCurrent=lCurr,
                                       # maxdLength=maxdLength,
                                       # areaPrevious=aPrev,
                                       # areaCurrent=aCurr,
                                       # maxdArea=maxdArea
                                       )
                if len(indMP) > 0:
                    trackNrP = trackNrPrev[indMP]
                    trackNrC = trackNrCurr[indMC]
                    for tNP, tNC in zip(trackNrP, trackNrC):
                        keeper.linkTracks(tNP, tNC)

            print("t: {} - {}, linked {} particles".format(endTime, startTime, len(indMP)))

    return dfTracks
