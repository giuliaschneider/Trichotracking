import os.path

import numpy as np
import pandas as pd
from trichotracking.dfmanip import (calcMovingAverages,
                     calcVelocity,
                     convertPxToMeter, calcPeaksSingle)
from trichotracking.iofiles import export_movie

from ._classifier import segment_filaments
from .trackmeta import Trackmeta


class Trackkeeper:
    """
    Class storing the track dataframe and providing meta data.



    """

    def __init__(self, df, pixelFile, meta):
        self.df = df
        self.pixelFile = pixelFile
        self.meta = meta
        self.startExp = 0
        self.endExp = np.max(self.df.frame.values)
        self.maxFrame = self.endExp
        self.maxTrack = self.df.trackNr.max()

    @classmethod
    def fromDf(cls, df, dfPixellist, pixelFile):
        meta = Trackmeta.fromScratch(df)
        if dfPixellist is not None:
            dfPixellist.to_pickle(pixelFile)
        return cls(df, pixelFile, meta)

    @classmethod
    def fromFiles(cls, tracksFile, pixelFile, trackMetaFile):
        dfTracks = pd.read_csv(tracksFile)
        meta = Trackmeta.fromFile(trackMetaFile)
        return cls(dfTracks, pixelFile, meta)

    def addColumnMeta(self, df_new):
        self.meta.addColumn(df_new)

    def getDfTracksMeta(self):
        return self.meta.getDf()

    def getStartTimes(self):
        return self.meta.getStartTimes()

    def getEndTimes(self):
        return self.meta.getEndTimes()

    def getStartTracks(self, t):
        return self.meta.getStartTracks(t)

    def getEndTracks(self, t):
        return self.meta.getEndTracks(t)

    def getMidTracks(self, startTime, endTime):
        return self.meta.getMidTracks(startTime, endTime )

    def setTrackStart(self, trackNr, newStartT):
        self.meta.setTrackStart(trackNr, newStartT)

    def setTrackEnd(self, trackNr, newEndT):
        self.meta.setTrackEnd(trackNr, newEndT)

    def getTracksAtTime(self, t, trackNrs=None):
        if trackNrs is not None:
            condition = ((self.df.frame == t)
                         & (self.df.trackNr.isin(trackNrs)))
        else:
            condition = (self.df.frame == t)
        return self.df[condition]

    def updateTrackNr(self, oldNr, newNr, t=None):
        """ Sets the oldNr to newNr, beginning from t. """
        if t is None:
            self.df.loc[(self.df.trackNr == oldNr), 'trackNr'] = newNr
        else:
            self.df.loc[((self.df.trackNr == oldNr)
                         & (self.df.frame >= t)), 'trackNr'] = newNr

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
        self.meta.setTrackEnd(trackNr, t - 1)

    def addMetaTrackType(self, dfAggMeta):
        single, aligned, cross = segment_filaments(self.df, dfAggMeta)
        aggregate = dfAggMeta[dfAggMeta.n > 2].trackNr.values
        self.meta.addTrackType(single, aligned, cross, aggregate)

    def getTrackNrPairs(self):
        return self.meta.getTrackNrPairs()

    def getTrackNrSingles(self):
        return self.meta.getTrackNrSingles()

    def getDfTracksComplete(self, trackNrs=None):
        dfPixellist = pd.read_pickle(self.pixelFile)
        if trackNrs is None:
            df = self.df.merge(dfPixellist, left_on='index', right_on='index')
        else:
            df = self.df[self.df.trackNr.isin(trackNrs)].merge(dfPixellist, left_on='index', right_on='index')
        return df

    def getDf(self):
        return self.df

    def setTime(self, times):
        self.df['time'] = times[self.df.frame]

    def setLabel(self):
        self.df['label'] = self.df.trackNr

    def smoothCentroidPosition(self, wsize=11):
        columns = ["cx_um", "cy_um"]
        ma_columns = ["cx_ma", "cy_ma"]
        if not all(x in self.df.keys() for x in ma_columns):
            self.df = calcMovingAverages(self.df, wsize, columns, ma_columns)

    def calcLengthVelocity(self, pxConversion):
        pxCols = ["length", "cx", "cy"]
        umCols = ["length_um", "cx_um", "cy_um"]
        if not all(x in self.df.keys() for x in umCols):
            self.df = convertPxToMeter(self.df, pxCols, umCols, pxConversion)
        self.smoothCentroidPosition()
        if not all(x in self.df.keys() for x in ['v', 'v_abs']):
            self.df = calcVelocity(self.df, 'cx_ma', 'cy_ma', 'time')
            self.df['v_abs'] = self.df.v.abs()

    def calcReversals(self):
        if not 'peaks' in self.df.keys():
            self.df = calcPeaksSingle(self.df, 'v')

    def save(self, trackFile, pixelFile, trackMetaFile):
        self.df.to_csv(trackFile)
        self.meta.save(trackMetaFile)

    def update_meta(self):
        self.meta.update(self.df)

    def saveAnimation(self, dataDir, destDir):
        nTracks = int(np.max(self.df.trackNr.values)) + 1
        export_movie(dataDir,
                     dfparticles=self.df,
                     nTracks=nTracks,
                     filename=os.path.join(destDir, 'animation.avi'))
