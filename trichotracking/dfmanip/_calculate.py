import numpy as np
import pandas as pd
import scipy.signal

from IPython.core.debugger import set_trace


def calcChangeInCol(df, columns, diff_columns):
    """ Calculates the change in overlap for all tracks. """
    time = df["time"].diff(periods=-1).values
    dtime = pd.to_timedelta(time, unit='s') / np.timedelta64(1, 's')
    df[diff_columns] = df[columns].diff(periods=-1).divide(dtime, axis=0)
    return df


def calcMovingAverage(df, label, column, ma_column, wsize):
    """ Calculates the moving average for each individual track. """
    df.loc[df.label==label, ma_column] = (df.loc[df.label==label, column]
                                .rolling(window=wsize, center=True).mean())
    return df


def calcMovingAverages(df, wsize, columns, ma_columns):
    """ Calculates the moving average for all track. """
    labels = np.unique(df.label)
    for col in ma_columns:
        df[col] = np.nan

    dfg = df.groupby('label')
    df[ma_columns] = dfg.apply(lambda x: x[columns].
                               rolling(window=wsize, center=True).mean())
    return df

def calcPeaks(df, col, p=0):
    """ Adds a column peaks to df, which is 1 if the value is a peak."""
    df["peaks"] = np.nan
    labels = np.unique(df.label)
    for label in labels:
        x = df[df.label==label][col].values
        ind = df[df.label==label].index.values
        #print("Label = {}, p = {}".format(label, p))
        # Local maxima
        pe, properties = scipy.signal.find_peaks(x, prominence=p)
        df.loc[df.index.isin(ind[pe]), 'peaks'] = 1
        # Local minima
        x = -x
        pe, properties = scipy.signal.find_peaks(x, prominence=p)
        df.loc[df.index.isin(ind[pe]), 'peaks'] = 1
    return df


def convertPxToMeter(df, columns, pxLength):
    """ Converts values in columns from pixel to lengths."""
    for col in columns:
        df[col] *= pxLength
    return df


def calcReversalPeriod(df, peakcond, slabels=None):
    if slabels is not None:
        dfpeaks = df[peakcond & (df.label.isin(slabels))].copy()
    else:
        dfpeaks = df[peakcond].copy()
    dfg = dfpeaks.groupby('label')
    dfpeaks['trev'] = dfg.time.transform(lambda x: x.diff())
    dfpeaks['trev']=dfpeaks.trev.dt.seconds
    df = df.join(dfpeaks.trev)
    return df
