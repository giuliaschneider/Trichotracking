import numpy as np
import scipy.signal

__all__ = ['calcPeaks', 'calcPeaksSingle', 'calcReversalPeriod']


def calcPeaks(df, col, p=0):
    """ Adds a column peaks to df, which is 1 if the value is a peak."""
    df["reversals"] = np.nan
    labels = np.unique(df.label)
    for label in labels:
        x = df[df.label == label][col].values
        ind = df[df.label == label].index.values
        # print("Label = {}, p = {}".format(label, p))
        # Local maxima
        pe = scipy.signal.find_peaks(x, prominence=p)[0]
        df.loc[df.index.isin(ind[pe]), 'reversals'] = 1
        # Local minima
        x = -x
        pe = scipy.signal.find_peaks(x, prominence=p)[0]
        df.loc[df.index.isin(ind[pe]), 'reversals'] = 1
    return df


def calcPeaksSingle(dflinked, cv):
    """Calculates the reversal period for single tracks based on vel (cv)."""
    v = dflinked[dflinked[cv].abs() >= 0.1][[cv, 'label']]
    v['reversals'] = ((v.label == v.label.shift())
                  & (np.sign(v[cv]).diff() < 0))
    dflinked = dflinked.join(v.reversals)
    return dflinked


def calcReversalPeriod(df, peakcond, slabels=None):
    if slabels is not None:
        dfpeaks = df[peakcond & (df.label.isin(slabels))].copy()
    else:
        dfpeaks = df[peakcond].copy()
    dfg = dfpeaks.groupby('label')
    dfpeaks['trev'] = dfg.time.transform(lambda x: x.diff())
    dfpeaks['trev'] = dfpeaks.trev.dt.seconds
    df = df.join(dfpeaks.trev)
    return df
