import numpy as np
import pandas as pd

from .metakeeper import Metakeeper


class Trackmeta(Metakeeper):
    """
    Class storing the track meta info and providing meta data.



    """

    def __init__(self, df):
        super().__init__(df)
        self.startExp = 0
        self.endExp = self.df.endTime.max()

    @classmethod
    def fromScratch(cls, df):
        df_tr = createDfTracksMeta(df)
        return cls(df_tr)

    def update(self, df):
        self.df = createDfTracksMeta(df)

    def getStartTimes(self):
        """ Returns all frames in which at least one track is starting. """
        sTimes = self.df[self.df != self.startExp].startTime
        sTimes = np.unique(sTimes[~np.isnan(sTimes)])
        sTimes = np.sort(sTimes)
        return sTimes

    def getEndTimes(self):
        """ Returns all frames in which at least one track is ending. """
        endTimes = self.df[self.df != self.endExp].endTime
        endTimes = np.unique(endTimes[~np.isnan(endTimes)])
        endTimes = np.sort(endTimes)
        return endTimes

    def getStartTracks(self, t):
        """ Returns trackNrs starting at time t. """
        return self.df[self.df.startTime == t].trackNr.values

    def getEndTracks(self, t):
        """ Returns trackNrs ending at time t. """
        return self.df[self.df.endTime == t].trackNr.values

    def getMidTracks(self, startTime, endTime):
        """ Returns trackNrs neither starting nor ending at times *t.
        """
        condition = ((self.df.startTime < startTime) & (self.df.endTime > endTime))
        return self.df[condition].trackNr.values

    def getTrackStart(self, trackNr):
        return self.df[self.df.trackNr == trackNr].startTime.values[0]

    def getTrackEnd(self, trackNr):
        return self.df[self.df.trackNr == trackNr].endTime.values[0]

    def getNFrames(self, trackNr):
        return self.df[self.df.trackNr == trackNr].nFrames.values[0]

    def setTrackStart(self, trackNr, newStartT):
        self.df.loc[self.df.trackNr == trackNr, 'startTime'] = newStartT

    def setTrackEnd(self, trackNr, newEndT):
        self.df.loc[self.df.trackNr == trackNr, 'endTime'] = newEndT

    def setNFrames(self, trackNr, nFrames):
        self.df.loc[self.df.trackNr == trackNr, 'nFrames'] = nFrames

    def addTrack(self, trackNr, startTime, endTime):
        nFrames = endTime - startTime
        self.df.loc[self.df.index.max() + 1] = [trackNr, startTime, endTime, nFrames, np.nan]

    def dropTrack(self, trackNr):
        """ Drops tracks trackNr from meta dataframe. """
        ind = self.df[self.df.trackNr == trackNr].index[0]
        self.df.drop(ind, inplace=True)

    def addTrackType(self, single, aligned, cross, aggregate):
        self.df.loc[self.df.trackNr.isin(single), 'type'] = 1
        self.df.loc[self.df.trackNr.isin(aligned), 'type'] = 2
        self.df.loc[self.df.trackNr.isin(cross), 'type'] = 3
        self.df.loc[self.df.trackNr.isin(aggregate), 'type'] = 4

    def getTrackNrPairs(self):
        return self.df[self.df.type == 2].trackNr.values

    def getTrackNrSingles(self):
        return self.df[self.df.type == 1].trackNr.values


def createDfTracksMeta(df):
    nTracks = df.trackNr.nunique()

    if nTracks == df.trackNr.size:
        df_tr = pd.DataFrame({'trackNr': df.trackNr})
        df_tr['startTime'] = df.frame
        df_tr['endTime'] = df.frame
        df_tr['nFrames'] = 1
        df_tr['length_mean'] = df.length
        df_tr.set_index('trackNr')
    else:
        dfg = df.groupby('trackNr', as_index=False)
        df_tr = pd.DataFrame({'trackNr': dfg.first().trackNr})
        df_tr['startTime'] = dfg.first().frame
        df_tr['endTime'] = dfg.last().frame
        df_tr['nFrames'] = dfg.count().area
        df_tr['length_mean'] = dfg.mean().length
        df_tr.set_index('trackNr')
    return df_tr
