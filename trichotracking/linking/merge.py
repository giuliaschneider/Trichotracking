import os
import numpy as np
import pandas as pd

from geometry import minDistMinBoxes, calcCenterOfMass
from trackkeeper import Trackkeeper
from utility import split_list

from .match import matcher


from IPython.core.debugger import set_trace


def merge(keeper):
    merger = Merger(keeper)
    df_merge = merger.getDfMerge()
    df_split = merger.getDfSplit()
    return df_merge, df_split





class dfMerge:

    def __init__(self):
        cols = ['Frame', 'trackNr', 'Track1', 'Track2']
        self.df_merge = pd.DataFrame(columns=cols)
        self.df_split = pd.DataFrame(columns=cols)

    def addTrack(self, df, time1, mt, t1, t2):
        ind = df.shape[0]+1
        df.loc[ind] = [time1, mt, t1, t2]

    def addTrackSplit(self, time1, mt, t1, t2):
        self.addTrack(self.df_split, time1, mt, t1, t2)

    def addTrackMerge(self, time1, mt, t1, t2):
        self.addTrack(self.df_merge, time1, mt, t1, t2)



class Merger:

    def __init__(self, 
                 keeper,
                 maxLinkTime=3,
                 maxMergeDistBox=7,
                 maxMergeDist=15):
        self.keeper = keeper
        self.mergekeeper = dfMerge()
        self.maxLinkTime = maxLinkTime
        self.maxMergeDistBox = maxMergeDistBox
        self.maxMergeDist = maxMergeDist
        
        self.maxTrack = self.keeper.maxTrack
        self.maxFrame = self.keeper.maxFrame

        self.iterate_times()


    def iterate_times(self):
        """ Iterates through all start- and endtimes and merge/split."""

        for i in range(1, self.maxLinkTime+1):
            self.endTimes = self.keeper.getEndTimes()
            self.startTimes = self.keeper.getStartTimes()-1
            self.times = np.unique(np.append(self.startTimes, self.endTimes))
            for endTime in self.times:
                self.startTime = endTime + i
                self.startTracks = self.keeper.getStartTracks(self.startTime)
                self.endTime = endTime
                self.endTracks = self.keeper.getEndTracks(self.endTime)
                self.split()
                self.merge()


    def split(self):
        """ Merges track segments by connecting ends to track midpoints. """

        # Splitting one ending track to to starting tracks
        mt1_t0, mt2_t0, mt_t1 = self.matchTwoOneTrackEnds(self.startTracks,
                                                          self.startTime, 
                                                          self.endTracks, 
                                                          self.endTime)
        self.updateDataFramesSplitEnds(self.endTime, mt1_t0, mt2_t0, mt_t1)

        # Merge one ending track to the middle of other track
        midTracks = self.keeper.getMidTracks(self.startTime, self.endTime)
        mt1_t0, mt2_t0, mt_t1 = self.matchOneEndOneMiddleTrack(
                self.startTracks, self.startTime, midTracks, self.endTime)
        self.updateDataFramesSplitMiddle(self.endTime, mt1_t0, mt2_t0, mt_t1)

    def merge(self):
        """ Merges track segments by connecting ends to track midpoints. """

        # Merge to ending tracks to one starting track
        mt1_t0, mt2_t0, mt_t1 = self.matchTwoOneTrackEnds(self.endTracks,
                                                          self.endTime, 
                                                          self.startTracks, 
                                                          self.startTime)
        self.updateDataFramesMergeEnds(self.startTime, mt1_t0, mt2_t0, mt_t1)

        # Merge one ending track to the middle of other track
        midTracks = self.keeper.getMidTracks(self.endTime, self.startTime)
        mt1_t0, mt2_t0, mt_t1 = self.matchOneEndOneMiddleTrack(
                self.endTracks, self.endTime, midTracks, self.startTime)
        self.updateDataFramesMergeMiddle(self.startTime, mt1_t0, mt2_t0, mt_t1)


    def matchTwoOneTrackEnds(self, tracks_t0, t0, tracks_t1, t1):
        """ matcher two track starts/ends at t0, to one track start/end t1.
        Arguments:
        tracks_t0 --    tracks starting / ending at t0
        t0 --           time step
        tracks_t1 --    tracks ending / starting at t1
        t1 --           time stepupdateDataFramesSplitEnds
        """

        # Check if at least two ending/starting tracks
        if (tracks_t0.size >= 2) and  (tracks_t1.size >= 1):
            dft0 = self.keeper.getTracksAtTime(t0, tracks_t0)
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

    def matchOneEndOneMiddleTrack(self, tracks_t0, t0, tracks_t1, t1):
        """ matcher one track start/end at t0, to one track not start/end t1.

        Arguments:
        tracks_t0 --    tracks starting / ending at t0
        t0 --           time step
        tracks_t1 --    tracks not ending / starting at t0, t1
        t1 --
        """
        if (tracks_t0.size > 0) and (tracks_t1.size > 0):
            dft1t0 = self.keeper.getTracksAtTime(t0, tracks_t0)
            dft2t0 = self.keeper.getTracksAtTime(t0, tracks_t1)
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

    def updateDataFramesMergeEnds(self, time1, mt1_t0, mt2_t0, mt_t1):
        """ Update self.df and self.df_tracks. """

        for mt, t1, t2 in zip(mt_t1, mt1_t0, mt2_t0):
            print("t = {}, merged {} with {}, new {}".format(
                    time1, t1, t2, mt))
            self.mergekeeper.addTrackMerge(time1, mt, t1, t2)
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

            self.mergekeeper.addTrackMerge(t, newTrack, t1, t2)
            self.keeper.splitTrack(t2, newTrack, t)
            self.updateEndTracks(t1)
            self.updateEndTracks(t2)
            self.updateStartTracks(newTrack)

    def updateDataFramesSplitEnds(self, time1, mt1_t0, mt2_t0, mt_t1):
        """ Update self.df and self.df_tracks. """
        for mt, t1, t2 in zip(mt_t1, mt1_t0, mt2_t0):
            print("t = {}, split {} into {} and {}".format(
                    time1, mt, t1, t2))
            self.mergekeeper.addTrackSplit(time1, mt, t1, t2)
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
            self.mergekeeper.addTrackSplit(time1, t2, t1, newTrack)
            self.keeper.splitTrack(t2, newTrack, self.startTime)
            self.updateStartTracks(t1)
            self.updateStartTracks(t2)
            self.updateEndTracks(newTrack)

    def calcDistanceBB(self, dfP, dfC):
        """ Returns distance matrix between the bounding boxes of dfP, dfC."""
        boxes1 = dfP['min_box'].values
        boxes2 = dfC['min_box'].values

        # Calculate the minmal distance between bounding boxes
        bdist = minDistMinBoxes(boxes1, boxes2)
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
            dft0 = self.keeper.df[self.keeper.df.frame==t0]
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
            dfm = self.keeper.getTracksAtTime(tm, mTracks)
        trackNrCurr = dfm.trackNr.values
        cxCurr = dfm.cx.values
        cyCurr = dfm.cy.values
        indMP, indMC = matcher(comx, comy, cxCurr, cyCurr, self.maxMergeDist,
                            indP1=tracks1, indP2=tracks2, indC=mTracks)
        return tracks1[indMP], tracks2[indMP], trackNrCurr[indMC]


    def updateStartTracks(self, trackNr):
        """ Removes trackNr from endTracks."""
        self.startTracks = self.startTracks[self.startTracks != trackNr]
        self.keeper.setTrackStart(trackNr, 0)


    def updateEndTracks(self, trackNr):
        """ Removes trackNr from endTracks."""
        self.endTracks = self.endTracks[self.endTracks != trackNr]
        self.keeper.setTrackEnd(trackNr, self.maxFrame)



    def getDfMerge(self):
        return self.mergekeeper.df_merge

    def getDfSplit(self):
        return self.mergekeeper.df_split