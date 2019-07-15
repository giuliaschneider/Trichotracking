import os.path
import numpy as np
import pandas as pd
from dfmanip.mdfagg import countFrames

from IPython.core.debugger import set_trace


def _iterate_tracks(tracks0, aggTracks, dfagg):
    tracks = []
    if (np.isnan(tracks0)).all():
        tracks = np.nan
    else:
        for t in tracks0:
            if t in aggTracks:
                newt = dfagg[dfagg.trackNr==t].tracks0.values[0]
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


class agg_keeper:

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


    def _listTracks(self, df, aggColumn):
        df[aggColumn]=df[['Track1', 'Track2']].values.tolist()
        df.drop(columns=['Track1', 'Track2'], inplace=True)
        return df



    def _fillTimes(self, row, col, df_tracks, col_tr):
        if not np.isnan(row[col]):
            return row[col]
        else:
            return df_tracks[df_tracks.trackNr==row.trackNr][col_tr].values[0]


    def _fillAggregation(self, row, dfagg, aggTracks):
        tracks0 = row.tracks0
        trackNr = row.trackNr
        for t in tracks0:
            if t in aggTracks:
                ind = dfagg[dfagg.trackNr==t].index[0]
                dfagg.at[ind, 'tracks1'] = [trackNr]


    def create_dfagg(self, df_tracks, dataDir, resultDir):
        """ Creates and saves a dataframe of aggregates.

        Positional parameters:
            df_merge --     Dataframe of merges
            df_split --     Dataframe of splits
            df_tracks --    Dataframe of all tracks
            dataDir --      directory of raw data
            resultDir --    directory of results


        Dataframe with columns:
            trackNr --      track number of aggregate track
            t0 --           start frame
            t1 --           end frame
            tracks0 --      trackNr before merged
            stracks0        trackNrs of single filaments before merge
            tracks1 --      trackNrs after split
            stracks1 --     tracksNrs of single filaments after split
            n0 --           number of tracks merging
            ns0 --          number of single filaments
            n1 --           number of tracks splitting
                            (nan: track ends without splitting,
                             1: track aggregates,
                             2: track splits)
            ns1 --          number of single filaments after
            n --            number of single filaments (max of ns0, ns1)
            breakup         breakup reason
                            1: aggregates
                            2: splits up
                            3: movie finished
                            4: unknown
        """
        cols = ['aggTrack', 't0', 't1', 'tracks0', 'tracks1', 'nfil', 'breakup']
        #aggTracks = np.unique(df_merge.trackNr.values)

        endExp = np.max(df_tracks.endTime.values)

        # Join df_merge and df_split, create dfagg
        self.df_merge = self._listTracks(self.df_merge, 'tracks0')
        self.df_merge.rename(columns={'Frame': 't0'}, inplace=True)
        self.df_merge.set_index('trackNr',inplace=True)
        self.df_split = self._listTracks(self.df_split, 'tracks1')
        self.df_split.rename(columns={'Frame': 't1'}, inplace=True)
        self.df_split.set_index('trackNr',inplace=True)
        dfagg = self.df_merge.join(self.df_split, how='outer')
        dfagg.reset_index(inplace=True)

        #Get single filaments starting
        aggTracks = dfagg.trackNr.values
        dfagg.sort_values(by='t0', inplace=True)
        dfagg['stracks0'] = dfagg.apply(lambda row:
                _getTracks(row, 'tracks0', aggTracks, dfagg), axis=1)
        dfagg.sort_values(by='t1', inplace=True)
        dfagg['stracks1'] = dfagg.apply(lambda row:
                _getTracks(row, 'tracks1', aggTracks, dfagg), axis=1)
        dfagg = countFrames(dfagg)

        # Get greater aggregates
        df=dfagg[(dfagg.n0 - dfagg.ns0).abs()>0].copy()
        df.apply(lambda row: self._fillAggregation(row, dfagg, aggTracks),
                 axis=1)

        dfagg['t0'] = dfagg.apply(lambda row:
                self._fillTimes(row,'t0', df_tracks, 'startTime'), axis=1)
        dfagg['t1'] = dfagg.apply(lambda row:
                self._fillTimes(row,'t1', df_tracks, 'endTime'), axis=1)


        dfagg = countFrames(dfagg)

        dfagg['breakup'] = 0
        dfagg.loc[dfagg.n1==1, 'breakup'] = 1
        dfagg.loc[dfagg.n1==2, 'breakup'] = 2
        dfagg.loc[(dfagg.breakup==0)&(dfagg.t1==endExp), 'breakup'] = 3
        dfagg.loc[dfagg.breakup==0, 'breakup'] = 4

        # Save df to text
        dir = os.path.basename(os.path.normpath(dataDir))
        filename = os.path.join(resultDir, dir+"_tracks_agg.txt")
        dfagg.to_csv(filename)

        return dfagg
