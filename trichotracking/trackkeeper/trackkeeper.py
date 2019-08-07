import os.path
import numpy as np
import pandas as pd

from iofiles import find_img, export_movie

from .trackmeta import Trackmeta

from IPython.core.debugger import set_trace


class Trackkeeper:
    """
    Class storing the track dataframe and providing meta data.



    """

    def __init__(self, df, dfPixellist, meta):
        self.df = df
        self.dfPixellist = dfPixellist
        self.meta = meta
        self.startExp = 0
        self.endExp = np.max(self.df.frame.values)
        self.maxFrame = self.endExp
        self.maxTrack = self.df.trackNr.max()

    @classmethod
    def fromDf(cls, df, dfPixellist):
        meta = Trackmeta.fromScratch(df)
        return cls(df, dfPixellist, meta)

    @classmethod
    def fromFiles(cls, tracksFile, pixelFile, trackMetaFile):
        dfTracks = pd.read_csv(tracksFile)
        dfPixellist = pd.read_pickle(pixelFile)
        meta = Trackmeta.fromFiles(dfTracks, trackMetaFile)
        return cls(dfTracks, dfPixellist, meta)

    def getDfTracksMeta(self):
        return self.meta.df_tr

    def getStartTimes(self):
        return self.meta.getStartTimes()

    def getEndTimes(self):
        return self.meta.getEndTimes()

    def getStartTracks(self, t):
        return self.meta.getStartTracks(t)

    def getEndTracks(self, t):
        return self.meta.getEndTracks(t)

    def getMidTracks(self, *t):
        return self.meta.getMidTracks(t)
    
    def setTrackStart(self, trackNr, newStartT):
        self.meta.setTrackStart(trackNr, newStartT)

    def setTrackEnd(self, trackNr, newEndT):
        self.meta.setTrackEnd(trackNr, newEndT)

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

    def splitTrack(self, trackNr, newTrackNr, t):
        """Renames track2 to newTrack from time t, updates df and df_tracks."""
        self.updateTrackNr(trackNr, newTrackNr, t)
        self.meta.addTrack(newTrackNr, t, self.meta.getTrackEnd(trackNr))
        self.meta.setTrackEnd(trackNr, self.meta.getTrackEnd(t-1))

    def addMetaTrackType(self, dfAggMeta):
        self.meta.addTrackType(dfAggMeta)

    def getTrackNrPairs(self):
        return self.meta.getTrackNrPairs()

    def getDfTracksComplete(self):
        return self.df.merge(self.dfPixellist, left_on='index', right_on='index')


    def save(self, trackFile, pixelFile, trackMetaFile):
        self.df.to_csv(trackFile)
        self.dfPixellist.to_pickle(pixelFile)
        self.meta.save(trackMetaFile)

    def saveAnimation(self, dataDir, destDir):
        nTracks = int(np.max(self.df.trackNr.values)) + 1
        export_movie(dataDir, 
                     dfparticles=self.df, 
                     nTracks=nTracks,              
                     filename=os.path.join(destDir,'animation.avi'))