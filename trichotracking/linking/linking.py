import os
import numpy as np
import pandas as pd
import multiprocessing as mp

from geometry import minDistMinBoxes, calcCenterOfMass
from trackkeeper import Trackkeeper
from utility import split_list


from ._df_agg import agg_keeper
from .match import matcher

from IPython.core.debugger import set_trace


def link(dfTracks):
    
    print("Start Linking")

    processes = mp.cpu_count()
    frames = split_list(dfTracks.frame.unique().tolist(), processes)
    dfs = [dfTracks[dfTracks.frame.isin(frames[i])] for i in range(processes)]

    pool = mp.Pool(processes=processes)
    results = [pool.apply_async(link_part, 
                args=(dfs[x],)) for x in range(processes)]
    
    pool.close()
    pool.join()

    dfs = [result.get() for result in results]
    dfTracks = pd.concat(dfs, ignore_index=True)
    dfTracks = link_part(ndfTracks)
    return dfTracks


def link_part(dfTracks,
              maxLinkTime=3,
              maxLinkDist=10,
              maxdLength=15,
              maxdArea=20):
    """ Links track segments by connecting ends to starts. """


    keeper = Trackkeeper(dfTracks)


    # Iterate through linking time steps
    for i in range(1, maxLinkTime+1):
        endTimes = keeper.getEndTimes()

    # Iterate through unique endtimes
        for endTime in endTimes:
            print("Linking time " + str(endTime))
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
                                        lengthPrevious=lPrev, 
                                        lengthCurrent=lCurr, 
                                        maxdLength=maxdLength,
                                        areaPrevious=aPrev, 
                                        areaCurrent=aCurr, 
                                        maxdArea=maxdArea)
                if len(indMP) > 0:
                    trackNrP = trackNrPrev[indMP]
                    trackNrC = trackNrCurr[indMC]
                    for tNP, tNC in zip(trackNrP, trackNrC):
                        keeper.linkTracks(tNP, tNC)

    return dfTracks


