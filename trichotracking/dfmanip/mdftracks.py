import numpy as np
import pandas as pd
from dfmeta import get_files, get_agginducer, get_aggchambers, get_chambers
from overlap_analysis._get_agg_inducer import get_aggregating

from ._label import calcLabel
from .mdfagg import getTimes
from IPython.core.debugger import set_trace

def attachTracks(dftracksfile, df_tracks, track1, track2):
    # Update df_track
    df_tracks.loc[df_tracks.trackNr == track1, 'endTime'] \
        = df_tracks.loc[df_tracks.trackNr == track2,'endTime'].values[0]
    df_tracks.loc[df_tracks.trackNr == track1, 'nFrames'] \
      += df_tracks.loc[df_tracks.trackNr==track2,'nFrames'].values[0]
    ind = df_tracks[df_tracks.trackNr==track2].index[0]
    df_tracks.drop(ind, inplace=True)
    df_tracks.to_csv(dftracksfile)
    return df_tracks

def splitTracks(dftracksfile, df_tracks, trackNr, newTrackNr, frame):
    #  Adds new track to df_tracks
    endTime = df_tracks.loc[df_tracks.trackNr==trackNr,'endTime'].values[0]
    nFrames = endTime-frame
    set_trace()
    df_tracks.loc[newTrackNr] = [newTrackNr,frame,endTime,nFrames, np.nan]
    df_tracks.loc[df_tracks.trackNr == trackNr, 'endTime'] = frame
    df_tracks.to_csv(dftracksfile)
    return df_tracks

def changeType(dftrackfile, df_tracks, trackNr, newType):
    df_tracks.loc[df_tracks.trackNr==trackNr, 'type'] = newType
    df_tracks.to_csv(dftrackfile)
    return df_tracks



def import_dftracks(dftrackfile):
    df = pd.read_csv(dftrackfile)
    cols = df.columns[df.columns.str.startswith('Unnamed')]
    df.drop(columns=cols, inplace=True)
    return df


def import_alldftracks(dfMeta, expLabels=None):

    df_list = []

    if expLabels is None:
        expLabels = dfMeta.exp.values
    for exp in expLabels:
        trackFiles, aggFiles, dftrackFiles, timesFiles = get_files(dfMeta, exp)
        agginducer = get_agginducer(dfMeta, exp)
        aggchambers = get_aggchambers(dfMeta, exp)
        chambers = get_chambers(dfMeta, exp)

        for trackFile, timeFile, chamber in zip(dftrackFiles, timesFiles, chambers):
            df_track_c = import_dftracks(trackFile)
            df_track_c['exp'] = exp
            df_track_c['chamber'] = chamber
            trackNrs = df_track_c.trackNr.values
            df_track_c["label"] = calcLabel(trackNrs, chamber, exp)
            df_track_c['aggregating'] = get_aggregating(df_track_c, agginducer, aggchambers)

            # Normalize time scale
            df_track_c = getTimes(df_track_c, timeFile, 'startTime', 'endTime')
            meant = df_track_c[df_track_c.type==1].t.mean()
            df_track_c['t_norm'] = df_track_c .t/ meant

            df_list.append(df_track_c)

        print("Experiment " + exp + ": Imported dftracks")


    df_tracks = pd.concat(df_list, axis=0, ignore_index=True)
    return df_tracks
