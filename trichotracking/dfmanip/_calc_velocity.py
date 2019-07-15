import numpy as np
import numpy.linalg as la
import pandas as pd


from IPython.core.debugger import set_trace


def calcVLabel(dflabel, ccx, ccy, ctime):
    time = dflabel[ctime].diff(periods=-1).values
    dtime = -pd.to_timedelta(time, unit='s') / np.timedelta64(1, 's')
    r = dflabel[[ccx, ccy]].diff(periods=-1)
    v = ((r[ccx]**2 + r[ccy]**2)**0.5) / dtime
    theta = np.arctan2(r[ccy], r[ccx])
    v = np.sign(theta)*v
    return v

def calcLongVelocity(df, ccx, ccy, ctime):
    """Calculates the velocitiy between centroid points (cols ccx, ccy)."""
    dfg = df.groupby('label')
    v = dfg.apply(lambda x: calcVLabel(x, ccx, ccy, ctime))
    vnew = v.reset_index().set_index('level_1').drop('label', axis=1)
    df['v'] = vnew
    return df


def calcTrackDistance(dflinked, df_tracks):
    dfg = dflinked.groupby('label')
    dfgf = dfg[['cx_ma', 'cy_ma']].first()
    dfgl = dfg[['cx_ma', 'cy_ma']].last()



    plt.loglog(dd[dd.aggregating].tt, dd[dd.aggregating].msd)
    plt.loglog(dd[~dd.aggregating].tt, dd[~dd.aggregating].msd)

    d = (dfgf - dfgl)
    d['dist'] = (d.cx_ma**2 + d.cy_ma**2)**0.5
    df_tracks = df_tracks.join(d.dist, on='label')
    df_tracks['dist_time'] = df_tracks.dist / df_tracks.t

    meanv = dfg['v_abs'].mean()
    df_tracks = df_tracks.join(meanv, on='label')
    df_tracks['v_dist'] = df_tracks.v_abs / df_tracks.dist_time
    return df_tracks

def calcPeaksSingle(dflinked, cv):
    """Calculates the reversal period for single tracks based on vel (cv)."""
    v = dflinked[dflinked[cv].abs()>=0.1][[cv, 'label']]
    v['peaks'] = ( (v.label == v.label.shift())
                  &(np.sign(v[cv]).diff()<0))
    dflinked = dflinked.join(v.peaks)
    return dflinked
