import os
import numpy as np
import pandas as pd

from geometry import minDistBondingBoxes, calcCenterOfMass

from ._df_agg import agg_keeper
from ._df_tracks import track_keeper
from .match import matcher

from IPython.core.debugger import set_trace



class linker:
    def __init__(self,
                 df,
                 listTimes,
                 dataDir,
                 resultDir,
                 maxLinkTime=4,
                 maxLinkDist=40,
                 maxMergeDist=30,
                 maxDl=30,
                 interrupt=None):

        """ Links, merges or splits tracks."""
        # Initalize
        print("Start Linking")
        self.dforg = df
        self.dfobj = track_keeper(df, listTimes, dataDir, resultDir)
        self.dfaggobj = agg_keeper()
        self.linkTime = maxLinkTime

        self.maxTrack = np.max(self.dfobj.df.trackNr.values)
        self.maxFrame= np.max(self.dfobj.df.frame.values)
        self.dataDir = dataDir
        self.resultDir = resultDir
        nFrames = self.dfobj.endExp - self.dfobj.startExp
        self.maxLinkDists = np.ones(nFrames) * maxLinkDist
        self.maxDls = np.ones(nFrames) * maxDl
        self.maxMergeDist = maxMergeDist
        if interrupt is not None:
            self.maxLinkDists[interrupt] = 3*maxLinkDist
            self.maxDls[interrupt] = 10*maxDl
        self.maxMergeDistBox = 7

        self.dfobj.update_DfTracks(0)
        self.link()

        self.dfobj.filter_intermittentTracks()
        self.dfobj.update_DfTracks(3)
        self.iterate_times()
        self.dfobj.update_DfTracks(0)

        self.dfagg = self.dfaggobj.create_dfagg(self.dfobj.df_tr,dataDir,resultDir)
        self.filTracks = self.dfobj.filterAllTracks(self.dfagg)
        self.dfobj.saveTrackToText()


    def link(self):
        """ Links track segments by connecting ends to starts. """

        # Iterate through linking time steps
        for i in range(1, self.linkTime+1):
            self.endTimes = self.dfobj.getEndTimes()

        # Iterate through unique endtimes
            for endTime in self.endTimes:
                self.startTime = endTime + i
                self.startTracks = self.dfobj.getStartTracks(self.startTime)
                self.endTime = endTime
                self.endTracks = self.dfobj.getEndTracks(self.endTime)
                ind = int(endTime)
                self.maxLinkDist = self.maxLinkDists[ind]
                self.maxDl = self.maxDls[ind]

                # Link based on distance and difference in length
                if (self.endTracks.size > 0) and (self.startTracks.size > 0):
                    dfP = self.dfobj.getTracksAtTime(self.endTracks,
                                                     self.endTime)
                    trackNrPrev = dfP.trackNr.values
                    cxPrev = dfP.cx.values
                    cyPrev = dfP.cy.values
                    lPrev = dfP.length.values

                    dfC = self.dfobj.getTracksAtTime(self.startTracks,
                                                     self.startTime)
                    trackNrCurr = dfC.trackNr.values
                    cxCurr = dfC.cx.values
                    cyCurr = dfC.cy.values
                    lCurr = dfC.length.values

                    indMP, indMC = matcher(cxPrev, cyPrev, cxCurr, cyCurr,
                            self.maxLinkDist, lPrev, lCurr, self.maxDl)
                    if len(indMP) > 0:
                        trackNrP = trackNrPrev[indMP]
                        trackNrC = trackNrCurr[indMC]
                        for tNP, tNC in zip(trackNrP, trackNrC):
                            print("t = {}, linked {} to {}, deleting {}"
                                .format(self.endTime, tNC, tNP, tNC))
                            self.dfobj.linkTracks(tNP, tNC)




    def iterate_times(self):
        """ Iterates through all starttimes and split."""
        # Iterate through linking time steps
        for i in range(1, self.linkTime+1):
        # Iterate through unique endtimes
            self.endTimes = self.dfobj.getEndTimes()
            self.startTimes = self.dfobj.getStartTimes()-1
            self.times = np.unique(np.append(self.startTimes, self.endTimes))
            for endTime in self.times:
                self.startTime = endTime + i
                self.startTracks = self.dfobj.getStartTracks(self.startTime)
                self.endTime = endTime
                self.endTracks = self.dfobj.getEndTracks(self.endTime)
                self.split()
                self.merge()


    def merge(self):
        """ Merges track segments by connecting ends to track midpoints. """

        # Merge to ending tracks to one starting track
        mt1_t0, mt2_t0, mt_t1 = self.matchTwoOneTrackEnds(self.endTracks,
                            self.endTime, self.startTracks, self.startTime)
        self.updateDataFramesMergeEnds(self.startTime, mt1_t0, mt2_t0, mt_t1)

        # Merge one ending track to the middle of other track
        midTracks = self.dfobj.getMidTracks(self.endTime, self.startTime)
        mt1_t0, mt2_t0, mt_t1 = self.matchOneEndOneMiddleTrack(
                self.endTracks, self.endTime, midTracks, self.startTime)
        self.updateDataFramesMergeMiddle(self.startTime, mt1_t0, mt2_t0, mt_t1)


    def split(self):
        """ Merges track segments by connecting ends to track midpoints. """

        # Splitting one ending track to to starting tracks
        mt1_t0, mt2_t0, mt_t1 = self.matchTwoOneTrackEnds(self.startTracks,
                        self.startTime, self.endTracks, self.endTime)
        self.updateDataFramesSplitEnds(self.endTime, mt1_t0, mt2_t0, mt_t1)

        # Merge one ending track to the middle of other track
        midTracks = self.dfobj.getMidTracks(self.startTime, self.endTime)
        mt1_t0, mt2_t0, mt_t1 = self.matchOneEndOneMiddleTrack(
                self.startTracks, self.startTime, midTracks, self.endTime)
        self.updateDataFramesSplitMiddle(self.endTime, mt1_t0, mt2_t0, mt_t1)


    def matchOneEndOneMiddleTrack(self, tracks_t0, t0, tracks_t1, t1):
        """ matcher one track start/end at t0, to one track not start/end t1.

        Arguments:
        tracks_t0 --    tracks starting / ending at t0
        t0 --           time step
        tracks_t1 --    tracks not ending / starting at t0, t1
        t1 --
        """
        if (tracks_t0.size > 0) and (tracks_t1.size > 0):
            dft1t0 = self.dfobj.getTracksAtTime(tracks_t0, t0)
            dft2t0 = self.dfobj.getTracksAtTime(tracks_t1, t0)
            # Calculate the distance of bounding boxess
            tracks1 = dft1t0.trackNr.values
            tracks2 = dft2t0.trackNr.values
            bdist = self.calcDistanceBB(dft1t0, dft2t0)
            indP, indC = np.where(bdist <= self.maxMergeDistBox)
        else:
            indP = []

        # Points that are close together
        if len(indP) > 0:
            tracks1_t0 = tracks1[indP]
            tracks2_t0 = tracks2[indC]
            mt1_t0, mt2_t0, mt_t1 = self.matchCOM(tracks1_t0, tracks2_t0,
                                                  t0, tracks_t1, t1)
        else:
            mt1_t0, mt2_t0, mt_t1 = [], [], []

        return mt1_t0, mt2_t0, mt_t1


    def matchTwoOneTrackEnds(self, tracks_t0, t0, tracks_t1, t1):
        """ matcher two track starts/ends at t0, to one track start/end t1.
        Arguments:
        tracks_t0 --    tracks starting / ending at t0
        t0 --           time step
        tracks_t1 --    tracks ending / starting at t1
        t1 --           time step
        """

        # Check if at least two ending/starting tracks
        if (tracks_t0.size >= 2) and  (tracks_t1.size >= 1):

            dft0 = self.dfobj.getTracksAtTime(tracks_t0, t0)
            tracks1 = dft0.trackNr.values
            tracks2 = dft0.trackNr.values
            # Calculate the distance of bounding boxess
            bdist = self.calcDistanceBB(dft0, dft0)
            # Set entries of lower triangular matrix larger than maxDist
            tril_ind = np.tril_indices(bdist.shape[0])
            bdist[tril_ind] = self.maxMergeDistBox + 1
            indP, indC = np.where(bdist <= self.maxMergeDistBox)
        else:
            indP = []

        # matcher points that are close together based on center of mass
        if len(indP) > 0:
            tracks1_t0 = tracks1[indP]
            tracks2_t0 = tracks2[indC]
            mt1_t0, mt2_t0, mt_t1 = self.matchCOM(tracks1_t0, tracks2_t0,
                                                  t0, tracks_t1, t1)
        else:
            mt1_t0, mt2_t0, mt_t1 = [], [], []

        return mt1_t0, mt2_t0, mt_t1


    def updateDataFramesMergeEnds(self, time1, mt1_t0, mt2_t0, mt_t1):
        """ Update self.df and self.df_tracks. """

        for mt, t1, t2 in zip(mt_t1, mt1_t0, mt2_t0):
            print("t = {}, merged {} with {}, new {}".format(
                    time1, t1, t2, mt))
            self.dfaggobj.addTrackMerge(time1, mt, t1, t2)
            self.updateEndTracks(t1)
            self.updateEndTracks(t2)
            self.updateStartTracks(mt)


    def updateDataFramesMergeMiddle(self, t, mt1_t0, mt2_t0, mt_t1):
        """ Update self.df and self.df_tracks. """

        for mt, t1, t2 in zip(mt_t1, mt1_t0, mt2_t0):
            self.maxTrack += 1
            newTrack = int(self.maxTrack)
            print("t = {}, merged {} with {}, new {}".format(
                    t, t1, t2, newTrack))

            self.dfaggobj.addTrackMerge(t, newTrack, t1, t2)
            self.dfobj.addTrack(t2, newTrack, t)
            self.updateEndTracks(t1)
            self.updateEndTracks(t2)
            self.updateStartTracks(newTrack)



    def updateDataFramesSplitEnds(self, time1, mt1_t0, mt2_t0, mt_t1):
        """ Update self.df and self.df_tracks. """
        for mt, t1, t2 in zip(mt_t1, mt1_t0, mt2_t0):
            print("t = {}, split {} into {} and {}".format(
                    time1, mt, t1, t2))
            self.dfaggobj.addTrackSplit(time1, mt, t1, t2)
            self.updateEndTracks(mt)
            self.updateStartTracks(t1)
            self.updateStartTracks(t2)


    def updateDataFramesSplitMiddle(self, time1, mt1_t0, mt2_t0, mt_t1):
        """ Update self.df and self.df_tracks. """
        for mt, t1, t2 in zip(mt_t1, mt1_t0, mt2_t0):
            self.maxTrack += 1
            newTrack = int(self.maxTrack)
            print("t = {}, split {} into {} and {}".format(
                    time1, t2, t1, newTrack))
            self.dfaggobj.addTrackSplit(time1, t2, t1, newTrack)
            self.dfobj.addTrack(t2, newTrack, self.startTime)
            self.updateStartTracks(t1)
            self.updateStartTracks(t2)
            self.updateEndTracks(newTrack)


    def calcDistanceBB(self, dfP, dfC):
        """ Returns distance matrix between the bounding boxes of dfP, dfC."""
        boxes1 = dfP[['bx', 'by', 'bw', 'bh']].values
        boxes2 = dfC[['bx', 'by', 'bw', 'bh']].values

        # Calculate the minmal distance between bounding boxes
        bdist = minDistBondingBoxes(boxes1, boxes2)
        return bdist


    def matchCOM(self, tracks1, tracks2, t0,  mTracks, tm,
                 dft0=None, dfm=None):
        """ Matches merging/splitting based on center of mass of two fil.

        Parameters:
            tracks1 --  trackNrs of filament 1 (duplicates possible)
            tracks2 --  trackNrs of filament 2 (duplicates possible)
            t1 --       time 1
            mTracks --  possible merged tracks
            tm --       time 2, merged

            """
        # Calculalte the center of mass for each track1 / track2 pair
        if dft0 is None:
            dft0 = self.dfobj.df[self.dfobj.df.frame==t0]
        comx, comy = np.zeros(tracks1.shape), np.zeros(tracks1.shape)
        for i, (t1, t2) in enumerate(zip(tracks1, tracks2)):
            r1 = dft0[dft0.trackNr == t1][['cx','cy']].values
            a1 = dft0[dft0.trackNr == t1].area.values
            r2 = dft0[dft0.trackNr == t2][['cx','cy']].values
            a2 = dft0[dft0.trackNr == t2].area.values
            try:
                comx[i], comy[i] = calcCenterOfMass(r1, a1, r2, a2)
            except:
                set_trace()

        # Calculate center of mass of merged particle current time
        if dfm is None:
            dfm = self.dfobj.getTracksAtTime(mTracks, tm)
        trackNrCurr = dfm.trackNr.values
        cxCurr = dfm.cx.values
        cyCurr = dfm.cy.values
        indMP, indMC = matcher(comx, comy, cxCurr, cyCurr, self.maxMergeDist,
                            indP1=tracks1, indP2=tracks2, indC=mTracks)
        return tracks1[indMP], tracks2[indMP], trackNrCurr[indMC]


    def updateStartTracks(self, trackNr):
        """ Removes trackNr from endTracks."""
        self.startTracks = self.startTracks[self.startTracks != trackNr]
        self.dfobj.updateDfTrackStart(trackNr, 0)


    def updateEndTracks(self, trackNr):
        """ Removes trackNr from endTracks."""
        self.endTracks = self.endTracks[self.endTracks != trackNr]
        self.dfobj.updateDfTrackEnd(trackNr, self.maxFrame)

    def getDf(self):
        return self.dfobj.df

    def getDfagg(self):
        return self.dfagg

    def getFilTracks(self):
        return self.filTracks

    def getDfTracks(self):
        return self.dfobj.df_tr
