import os.path
import numpy as np
import pandas as pd

from iofiles import find_img, export_movie
from segmentation import segment_filaments
from IPython.core.debugger import set_trace

from .trackmeta import Trackmeta

class Trackkeeper:
    """
    Class storing the track dataframe and providing meta data.



    """

    def __init__(self, df):
        self.df = df
        self.meta = Trackmeta(df)
        self.startExp = 0
        self.endExp = np.max(self.df.frame.values)

    @classmethod
    def fromDf(cls, df):
        return cls(df)

    @classmethod
    def fromFiles(cls, tracksFile, pixelFile):
        dfTracks = pd.read_csv(tracksFile)
        dfPixellist = pd.read_pickle(pixelFile)
        dfTracks.merge(dfPixellist, left_on='index', right_on='index')
        return cls(dfTracks)

    def getStartTimes(self):
        return self.meta.getStartTimes()

    def getEndTimes(self):
        return self.meta.getEndTimes()

    def getStartTracks(self, t):
        return self.meta.getStartTracks(t)

    def getEndTracks(self, t):
        return self.meta.getEndTracks(t)

    def getTracksAtTime(self, t, trackNrs=None):
        if trackNrs is not None:
            condition = ((self.df.frame == t)
                        &(self.df.trackNr.isin(trackNrs)))
        else:
            condition = (self.df.frame == t)
        return self.df[condition]

    def updateTrackNr(self, oldNr, newNr, t=None):
        """ Sets the oldNr to newNr, beginning from t. """
        if t is None:
            self.df.loc[(self.df.trackNr==oldNr), 'trackNr'] = newNr
        else:
            self.df.loc[((self.df.trackNr==oldNr)
                       &(self.df.frame>=t)), 'trackNr'] = newNr

    def linkTracks(self, trackNr1, trackNr2):
        """ Links track2 to track2, updates df and df_tracks."""
        # Set trackNr2 to 1
        self.updateTrackNr(trackNr2, trackNr1)
        self.meta.setTrackEnd(trackNr1, self.meta.getTrackEnd(trackNr2))
        self.meta.setNFrames(trackNr1, 
                self.meta.getNFrames(trackNr1) + self.meta.getNFrames(trackNr2))
        self.meta.dropTrack(trackNr2)



    """
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





    def addTrack(self, track2, newTrack, t):
        # Renames track2 to newTrack from time t, updates df and df_tracks.
        # Update df
        self.updateTrackNr(track2, newTrack, t)

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
        """
    """ self.df_tr = self.df_tr[((self.df_tr.trackNr.isin(aggTracks))
                      |(self.df_tr.trackNr.isin(fsingleTracks)))]
        self.df = self.df[((self.df.trackNr.isin(aggTracks))
                         |(self.df.trackNr.isin(fsingleTracks)))]"""
    """
        try:
            self.df_tr['startTime'] = self.listTimes[self.df_tr.startTime]
            self.df_tr['endTime'] = self.listTimes[self.df_tr.endTime]
            self.df_tr.loc[self.df_tr.trackNr.isin(fsingleTracks), 'type']=1
            self.df_tr.loc[self.df_tr.trackNr.isin(filAlignedTracks), 'type']=2
            self.df_tr.loc[self.df_tr.trackNr.isin(filCrossTracks), 'type']=3
            self.df_tr.loc[self.df_tr.trackNr.isin(severalFilTracks), 'type']=4
        except:
            set_trace()
        return filAlignedTracks
    """

    """
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
        
        self.listTimes = np.asarray(self.listTimes)
        filename = os.path.join(self.resultDir, "times.csv")
        np.savetxt(filename, self.listTimes)

        self.saveDfTracks()

        # Save tracking animation
        self.nTracks = int(np.max(self.df.trackNr.values)) + 1
        self.listOfFiles = find_img(self.dataDir)
        export_movie(self.dataDir, dfparticles=self.df, nTracks=self.nTracks,              
                     filename=os.path.join('results','tracked.avi'))

    """