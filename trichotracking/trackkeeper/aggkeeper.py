import numpy as np
import pandas as pd

from dfmanip import columnsToListColumns, combineNanCols, listToColumns
from .metakeeper import Metakeeper
from pdb import set_trace


class Aggkeeper(Metakeeper):
    """ 
    Class storing and providing information about aggregate tracks.

    Dataframe with columns:
    trackNr --      track number of aggregate track
    t0 --           start frame
    t1 --           end frame
    tracks0 --      trackNr before merged
    stracks0        trackNrs of single filaments before merge
    tracks1 --      trackNrs after split
    stracks1 --     tracksNrs of single filaments after split
    mTrack1 --      trackNr of one track merged or split
    mTrack2 --      trackNr of other track merged or split
    n --            number of single filaments
    breakup         breakup reason
                    1: aggregates
                    2: splits up
                    3: movie finished
                    4: unknown
"""

    def __init__(self, dfagg):
        self.df = dfagg

    @classmethod
    def fromScratch(cls, df_merge, df_split, trackkeeper):
        """ Creates meta dataframe of aggregates.

        Positional parameters:
            df_merge --     Dataframe of merges
            df_split --     Dataframe of splits
            df_tracks --    Dataframe of all tracks

        """

        endExp = trackkeeper.endExp

        # Join df_merge and df_split, create dfagg
        df_merge = columnsToListColumns(df_merge, 'tracks0', ['Track1', 'Track2'])
        df_merge.rename(columns={'Frame': 't0'}, inplace=True)
        df_merge.set_index('trackNr', inplace=True)
        df_split = columnsToListColumns(df_split, 'tracks1', ['Track1', 'Track2'])
        df_split.rename(columns={'Frame': 't1'}, inplace=True)
        df_split.set_index('trackNr', inplace=True)
        dfagg = df_merge.join(df_split, how='outer')
        dfagg.reset_index(inplace=True)

        # Get single filaments starting
        aggTracks = dfagg.trackNr.values
        dfagg.sort_values(by='t0', inplace=True)
        dfagg['stracks0'] = dfagg.apply(lambda row:
                                        _getTracks(row, 'tracks0', aggTracks, dfagg), axis=1)
        dfagg.sort_values(by='t1', inplace=True)
        dfagg['stracks1'] = dfagg.apply(lambda row:
                                        _getTracks(row, 'tracks1', aggTracks, dfagg), axis=1)
        dfagg = countFrames(dfagg)

        # Get greater aggregates
        df = dfagg[(dfagg.n0 - dfagg.ns0).abs() > 0].copy()
        df.apply(lambda row: fillAggregation(row, dfagg, aggTracks),
                 axis=1)

        dfTracksMeta = trackkeeper.getDfTracksMeta()
        dfagg['t0'] = dfagg.apply(lambda row:
                                  fillTimes(row, 't0', dfTracksMeta, 'startTime'), axis=1)
        dfagg['t1'] = dfagg.apply(lambda row:
                                  fillTimes(row, 't1', dfTracksMeta, 'endTime'), axis=1)

        dfagg = countFrames(dfagg)

        dfagg['breakup'] = 0
        dfagg.loc[dfagg.n1 == 1, 'breakup'] = 1
        dfagg.loc[dfagg.n1 == 2, 'breakup'] = 2
        dfagg.loc[(dfagg.breakup == 0) & (dfagg.t1 == endExp), 'breakup'] = 3
        dfagg.loc[dfagg.breakup == 0, 'breakup'] = 4

        # Create column which has either trackNr before merge or after split
        dfagg = combineNanCols(dfagg, 'deftracks', 'tracks0', 'tracks1')
        # Split column
        dfagg = listToColumns(dfagg, 'deftracks', ['mTrack1', 'mTrack2'])
        dfagg.drop(columns=['deftracks', 'n0', 'n1', 'ns0', 'ns1'], inplace=True)

        return cls(dfagg)





def fillTimes(row, col, df_tracks, col_tr):
    if not np.isnan(row[col]):
        return row[col]
    else:
        return df_tracks[df_tracks.trackNr == row.trackNr][col_tr].values[0]


def fillAggregation(row, dfagg, aggTracks):
    tracks0 = row.tracks0
    trackNr = row.trackNr
    for t in tracks0:
        if t in aggTracks:
            ind = dfagg[dfagg.trackNr == t].index[0]
            dfagg.at[ind, 'tracks1'] = [trackNr]


def _iterate_tracks(tracks0, aggTracks, dfagg):
    tracks = []
    if (np.isnan(tracks0)).all():
        tracks = np.nan
    else:
        for t in tracks0:
            if t in aggTracks:
                newt = dfagg[dfagg.trackNr == t].tracks0.values[0]
                tr0 = _iterate_tracks(newt, aggTracks, dfagg)
                if (np.isnan(tr0)).all():
                    tracks.append(t)
                else:
                    tracks += tr0
            else:
                tracks.append(t)
    return tracks


def _getTracks(row, col, aggTracks, dfagg):
    tracks0 = row[col]
    if (np.isnan(tracks0)).all():
        tracks = tracks0
    else:
        tracks = _iterate_tracks(tracks0, aggTracks, dfagg)
    return tracks


def countFrames(dfagg):
    dfagg['n0'] = dfagg.tracks0.str.len()
    dfagg['ns0'] = dfagg.stracks0.str.len()
    dfagg.sort_values(by='trackNr', inplace=True)
    dfagg['n1'] = dfagg.tracks1.str.len()
    dfagg['ns1'] = dfagg.stracks1.str.len()
    dfagg['n'] = dfagg[['ns0', 'ns1']].max(axis=1)
    return dfagg
