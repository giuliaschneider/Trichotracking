import numpy as np
import pandas as pd

class Trackmeta:
    """
    Class storing the track meta info and providing meta data.



    """


    def __init__(self, df):
        self.df = df
        self.startExp = self.df.frame.min()
        self.endExp = self.df.frame.max()

        self.df_tr = self.createDfTracksMeta()


    def createDfTracksMeta(self):
        nTracks = self.df.trackNr.nunique()

        if nTracks == self.df.trackNr.size:
            self.df_tr = pd.DataFrame({'trackNr': self.df.trackNr})
            self.df_tr['startTime'] = self.df.frame
            self.df_tr['endTime'] = self.df.frame
            self.df_tr['nFrames'] = 1
            self.df_tr.set_index('trackNr')
        else:
            dfg = self.df.groupby('trackNr', as_index=False)
            self.df_tr = pd.DataFrame({'trackNr': dfg.first().trackNr})
            self.df_tr['startTime'] = dfg.first().frame
            self.df_tr['endTime'] = dfg.last().frame
            self.df_tr['nFrames'] = dfg.count().area
            self.df_tr.set_index('trackNr')
        return self.df_tr

    def getStartTimes(self):
        """ Returns all frames in which at least one track is starting. """
        sTimes = self.df_tr[self.df_tr!=self.startExp].startTime
        sTimes = np.unique(sTimes[~np.isnan(sTimes)])
        sTimes = np.sort(sTimes)
        return sTimes

    def getEndTimes(self):
        """ Returns all frames in which at least one track is ending. """
        endTimes = self.df_tr[self.df_tr!=self.endExp].endTime
        endTimes = np.unique(endTimes[~np.isnan(endTimes)])
        endTimes = np.sort(endTimes)
        return endTimes

    def getStartTracks(self, t):
        """ Returns trackNrs starting at time t. """
        return self.df_tr[self.df_tr.startTime==t].trackNr.values

    def getEndTracks(self, t):
        """ Returns trackNrs ending at time t. """
        return self.df_tr[self.df_tr.endTime==t].trackNr.values

    def getMidTracks(self, *t):
        """ Returns trackNrs neither starting nor ending at times *t. """
        condition = ((self.df_tr.starttime.isin(t))
                    &(self.df_tr.endtime.isin(t)))
        return self.df_tr[condition].trackNr.values

    def getTrackStart(self, trackNr):
        return self.df_tr[self.df_tr.trackNr==trackNr].startTime

    def getTrackEnd(self, trackNr):
        return self.df_tr[self.df_tr.trackNr==trackNr].endTime

    def getNFrames(self, trackNr):
        return self.df_tr[self.df_tr.trackNr==trackNr].nFrames

    def setTrackStart(self, trackNr, newStartT):
        self.df_tr.loc[self.df_tr.trackNr==trackNr, 'startTime'] = newStartT

    def setTrackEnd(self, trackNr, newEndT):
        self.df_tr.loc[self.df_tr.trackNr==trackNr, 'endTime'] = newEndT

    def setNFrames(self, trackNr, nFrames):
        self.df_tr.loc[self.df_tr.trackNr==trackNr, 'nFrames'] = nFrames


    def dropTrack(self, trackNr):
        """ Drops tracks trackNr from meta dataframe. """
        ind = self.df_tr[self.df_tr.trackNr==trackNr].index[0]
        self.df_tr.drop(ind, inplace=True)
