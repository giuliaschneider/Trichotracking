import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace


def dropTrack(aggFile, dfagg, trackNr ):
    if trackNr in dfagg.trackNr:
        ind = dfagg[dfagg.trackNr==trackNr].index[0]
        dfagg.drop(ind, axis=1, inplace=True)
        dfagg.to_csv(aggFile)
    return dfagg




def addTrack(aggFile, dfagg, dftracksfile, df_tracks, trackNr, tracks0,
             tracks1, breakup):
    if "Unnamed: 0" in dfagg.columns:
        dfagg.drop('Unnamed: 0', axis=1, inplace=True)
    cols = dfagg.columns
    t0 = df_tracks[df_tracks.trackNr==trackNr].startTime.values[0]
    t1 = df_tracks[df_tracks.trackNr==trackNr].endTime.values[0]

    df_new = pd.DataFrame({'trackNr': [trackNr],'t0': [t0],'t1': [t1]})
    df_new['tracks0'] = [tracks0]
    df_new['tracks1'] = [tracks1]
    df_new['stracks0'] = [tracks0]
    df_new['stracks1'] = [tracks1]
    df_new = countFrames(df_new)
    df_new['breakup'] = breakup
    dfagg = dfagg.append(df_new, sort=False)
    dfagg.sort_values(by='trackNr', inplace=True)
    dfagg.to_csv(aggFile)

    df_tracks.loc[df_tracks.trackNr==trackNr, 'type']=2
    df_tracks.to_csv(dftracksfile)
    return dfagg

def getTimes(dfagg, timesFile, t0='t0', t1='t1'):
    times = np.loadtxt(timesFile)
    maxFrame = times.size
    t0_frame = dfagg[dfagg[t0]<maxFrame][t0].values.astype('int')
    t1_frame = dfagg[dfagg[t1]<maxFrame][t1].values.astype('int')

    dfagg.loc[dfagg[t0]<maxFrame, t0] = times[t0_frame]
    dfagg.loc[dfagg[t1]<maxFrame, t1] = times[t1_frame]
    dfagg['t'] = dfagg[t1] - dfagg[t0]
    return dfagg
