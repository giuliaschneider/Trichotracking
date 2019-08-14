import numpy as np
import pandas as pd
from .metakeeper import Metakeeper


class Pairkeeper(Metakeeper):
    """
    Class storing and providing information about filament pair tracks.

    Dataframe with columns:
    trackNr --      track number of filament pair track
    length1 --      length of longer filament in px
    length2 --      length of shorter filament in px
    breakup         breakup reason
                    1: aggregates
                    2: splits up
                    3: movie finished
                    4: unknown
    """

    def __init__(self, df):
        super().__init__(df)

    @classmethod
    def fromScratch(cls, dfAggMeta, dfTracksMeta, pairTrackNrs):
        filTracks, filLengths, breakup = getFilLengthsBreakup(pairTrackNrs,
                                                              dfAggMeta,
                                                              dfTracksMeta)
        filLengths.sort(axis=1)
        filLengths = filLengths[:, ::-1]
        df = pd.DataFrame({'trackNr': filTracks,
                           'length1': filLengths[:, 0],
                           'length2': filLengths[:, 1],
                           'breakup': breakup,
                           'type': np.nan})
        return cls(df)

    def getTrackNr(self):
        return self.df.trackNr.values

    def getLengths(self,):
        return self.df[['length1', 'length2']].values

    def save(self, file):
        self.df.to_csv(file)


def getFilLengthsBreakup(pairTrackNrs, dfagg, dfg):
    """ Get aggregate trackNr & lenghts of single filaments. """

    # Merge dfagg and dfg
    vdfg = dfg[['trackNr', 'length_mean']]  # view of dfg
    props = {'how': 'left', 'right_on': 'trackNr'}
    suffix = ('0', '1')
    dfagg = pd.merge(dfagg, vdfg, **props, left_on='mTrack1', suffixes=suffix)
    suffix = ('1', '2')
    dfagg = pd.merge(dfagg, vdfg, **props, left_on='mTrack2', suffixes=suffix)

    cols = ['trackNr0', 'length_mean1', 'length_mean2', 'breakup']
    dftracks = dfagg[dfagg.trackNr0.isin(pairTrackNrs)][cols]
    tracks = dftracks.trackNr0.values
    filLengths = dftracks[['length_mean1', 'length_mean2']].values
    breakup = dftracks.breakup.values
    return tracks, filLengths, breakup
