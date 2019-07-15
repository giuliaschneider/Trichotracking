import numpy as np
import pandas as pd

from ._calc_velocity import calcLongVelocity, calcPeaksSingle
from ._calculate import convertPxToMeter, calcMovingAverages
from .mdfagg import dropTrack
from .mdftracks import attachTracks, splitTracks

from IPython.core.debugger import set_trace

def attachTracksRaw(linkedfile, df, dftracksfile, df_tracks, aggFile, dfagg,
                    track1, track2):
    """ Attaches track2 to track1. """
    df.loc[df.trackNr==track2, 'trackNr'] = track1
    df.to_csv(linkedfile)
    df_tracks= attachTracks(dftracksfile, df_tracks, track1, track2)
    dfagg = dropTrack(aggFile, dfagg, track2)
    return df, df_tracks, dfagg

def splitTracksRaw(linkedfile, df, dftracksfile, df_tracks, trackNr,
                      newTrackNr, frame):
    df.loc[(df.trackNr==trackNr) & (df.frame>frame), 'trackNr'] = newTrackNr
    df.to_csv(linkedfile)
    df_tracks = splitTracks(dftracksfile, df_tracks, trackNr, newTrackNr, frame)
    return df, df_tracks

def postprocess(linkedFile, timesFile, pxLength):
    df = pd.read_csv(linkedFile)
    cols = df.columns[df.columns.str.startswith('Unnamed')]
    df.drop(columns=cols, inplace=True)

    columns = ["length", "cx","cy"]
    df = convertPxToMeter(df, columns, pxLength)
    df['label'] = df.trackNr.astype('int')
    columns = ["cx", "cy"]
    ma_columns =  ["cx_ma","cy_ma"]
    df = calcMovingAverages(df, 13, columns, ma_columns)
    times = np.loadtxt(timesFile)
    df['time'] = times[df.frame]
    #df.time = pd.to_datetime(df.time, format='%Y-%m-%d %H:%M:%S.%f')

    df = calcLongVelocity(df, 'cx_ma', 'cy_ma', 'time')
    df['v_abs'] = df.v.abs()

    df = calcPeaksSingle(df, 'v')
    return df
