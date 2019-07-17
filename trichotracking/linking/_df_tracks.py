import os.path
import numpy as np
import pandas as pd

from iofiles import find_img, export_movie
from segmentation import segment_filaments
from plot import TrackingAnimation
from IPython.core.debugger import set_trace


class track_keeper:

    def __init__(self, df, listTimes, dataDir, resultDir):
        self.df = df
        self.listTimes = listTimes
        self.dataDir = dataDir
        self.resultDir = resultDir
        self.df.sort_values(by=['trackNr', 'frame'], inplace=True)
        self.startExp = 0
        self.endExp = np.max(self.df.frame.values)

    def update_DfTracks(self, minFrames):
        # Create meta dataframe
        dfg = self.df.groupby('trackNr', as_index=False)
        self.df_tr = pd.DataFrame({'trackNr': dfg.first().trackNr})
        self.df_tr['startTime'] = dfg.first().frame
        self.df_tr['endTime'] = dfg.last().frame
        self.df_tr['nFrames'] = dfg.count().area
        self.df_tr = self.df_tr[self.df_tr.nFrames>minFrames]
        self.df_tr.set_index('trackNr')

    def drop_trackNr(self, df, trackNr):
        """ Drops track trackNr from df. """
        ind = df[df.trackNr==trackNr].index[0]
        df.drop(ind, inplace=True)

    def filter_intermittentTracks(self):
        tracks = np.unique(self.df.trackNr)
        self.df['frame_diff'] = np.nan
        for t in tracks:
            self.df.loc[self.df.trackNr==t, 'frame_diff'] \
                = self.df[self.df.trackNr==t].frame.diff()
        dfg_frame = self.df.groupby('trackNr').mean().frame_diff
        tracks = dfg_frame[dfg_frame < 1.3].index.values
        #tracks = dfg_frame[dfg_frame == 1].index.values
        self.df = self.df[self.df.trackNr.isin(tracks)].copy()

    def filter_eccentricity(self):
        # Only keep tracks with high eccentricity
        dfg = self.df.groupby('trackNr', as_index=False)
        grouped = dfg.mean()
        tricho_tracks = grouped[grouped.eccentricity > 0.9].index
        self.df = self.df[self.df.trackNr.isin(tricho_tracks)].copy()

    def getEndTimes(self):
        endTimes = self.df_tr[self.df_tr!=self.endExp].endTime
        endTimes = np.unique(endTimes[~np.isnan(endTimes)])
        endTimes = np.sort(endTimes)
        return endTimes

    def getStartTimes(self):
        sTimes = self.df_tr[self.df_tr!=self.startExp].startTime
        sTimes = np.unique(sTimes[~np.isnan(sTimes)])
        sTimes = np.sort(sTimes)
        return sTimes

    def getStartTracks(self, t):
        """ Returns trackNrs starting at time t. """
        return self.df_tr[self.df_tr.startTime==t].trackNr.values

    def getEndTracks(self, t):
        """ Returns trackNrs ending at time t. """
        return self.df_tr[self.df_tr.endTime==t].trackNr.values

    def getMidTrackst(self, t0):
        startTracks = self.getStartTracks(t0)
        endTracks = self.getEndTracks(t0)
        tracks = np.unique(self.df_tr.trackNr)
        condition = ((self.df.frame==t0)
                    &(self.df.trackNr.isin(tracks))
                    &(~self.df.trackNr.isin(startTracks))
                    &(~self.df.trackNr.isin(endTracks)))
        midTracks = self.df[condition].trackNr.values
        return midTracks

    def getMidTracks(self, t0, t1):
        """ Returns the tracks that are neither starting/ending in t0/t1. """
        midtracks0 = self.getMidTrackst(t0)
        midtracks1 = self.getMidTrackst(t1)
        midtracks = np.intersect1d(midtracks0, midtracks1)
        return midtracks


    def getTracksAtTime(self, tracks, t):
        return self.df[ (self.df.frame == t)
                       &(self.df.trackNr.isin(tracks)) ]


    def updateDfTrackStart(self, trackNr, newStartTime):
        self.df_tr.loc[self.df_tr.trackNr==trackNr, 'startTime'] = \
                                                            newStartTime

    def updateDfTrackEnd(self, trackNr, newEndTime):
        self.df_tr.loc[self.df_tr.trackNr==trackNr, 'endTime'] = \
                                                            newEndTime

    def setDFTrackNr(self, oldNr, newNr, t=None):
        """ Sets the oldNr to newNr, beginning from t. """
        if t is None:
            self.df.loc[(self.df.trackNr==oldNr), 'trackNr'] = newNr
        else:
            self.df.loc[((self.df.trackNr==oldNr)
                       &(self.df.frame>=t)), 'trackNr'] = newNr

    def linkTracks(self, track1, track2):
        """ Links track2 to track2, updates df and df_tracks."""
        # Update df
        self.setDFTrackNr(track2, track1)
        # Update df_track
        self.df_tr.loc[self.df_tr.trackNr == track1, 'endTime'] \
            = self.df_tr.loc[self.df_tr.trackNr == track2,'endTime'].values[0]
        self.df_tr.loc[self.df_tr.trackNr == track1, 'nFrames'] \
          += self.df_tr.loc[self.df_tr.trackNr==track2,'nFrames'].values[0]

        self.drop_trackNr(self.df_tr, track2)


    def addTrack(self, track2, newTrack, t):
        """ Renames track2 to newTrack from time t, updates df and df_tracks."""
        # Update df
        self.setDFTrackNr(track2, newTrack, t)

        #  Adds new track to df_tracks
        endTime = self.df_tr.loc[self.df_tr.trackNr == track2,
                                'endTime'].values[0]
        nFrames = endTime-t
        self.df_tr.loc[newTrack] = [newTrack,t,endTime,nFrames]

        # Updates mid track in df_tracks
        self.df_tr.loc[self.df_tr.trackNr==track2, 'endTime'] = t-1


    def filterAllTracks(self, dfagg):
        fsingleTracks, filAlignedTracks, filCrossTracks = \
                                    segment_filaments(self.df, dfagg)
        severalFilTracks = dfagg[dfagg.n>2].trackNr.values
        """self.df_tr = self.df_tr[((self.df_tr.trackNr.isin(aggTracks))
                      |(self.df_tr.trackNr.isin(fsingleTracks)))]
        self.df = self.df[((self.df.trackNr.isin(aggTracks))
                         |(self.df.trackNr.isin(fsingleTracks)))]"""

        self.df_tr['startTime'] = self.listTimes[self.df_tr.startTime]
        self.df_tr['endTime'] = self.listTimes[self.df_tr.endTime]
        self.df_tr.loc[self.df_tr.trackNr.isin(fsingleTracks), 'type']=1
        self.df_tr.loc[self.df_tr.trackNr.isin(filAlignedTracks), 'type']=2
        self.df_tr.loc[self.df_tr.trackNr.isin(filCrossTracks), 'type']=3
        self.df_tr.loc[self.df_tr.trackNr.isin(severalFilTracks), 'type']=4
        return filAlignedTracks


    def getDfTracks(self):
        return self.df_tr

    def saveDfTracks(self, ):
        #dir = os.path.basename(os.path.normpath(self.dataDir))
        # Save df_tr to text
        filename = os.path.join(self.resultDir, "df_tracks.csv")
        self.df_tr.to_csv(filename)


    def saveTrackToText(self):
        # Save df to text
        #dir = os.path.basename(os.path.normpath(self.dataDir))
        filename = os.path.join(self.resultDir, "tracks.csv")
        self.df.to_csv(filename)

        self.saveDfTracks()

        # Save tracking animation
        self.nTracks = int(np.max(self.df.trackNr.values)) + 1
        self.listOfFiles = find_img(self.dataDir)
        export_movie( self.dataDir, dfparticles=self.df, nTracks=self.nTracks)

