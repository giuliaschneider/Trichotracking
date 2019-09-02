import numpy as np
from IPython.terminal.debugger import set_trace

__all__ = ['calcChangeInTime', 'calcVelocity', 'calcSingleFilamentVelocity']


def calcDTime(df, ctime):
    time = df[ctime].diff(periods=-1).values
    # dtime = pd.to_timedelta(time, unit='s') / np.timedelta64(1, 's')
    return time


def calcChangeInTime(df, ctime, columns, diff_columns):
    """ Calculates the change in overlap for all tracks. """
    dtime = calcDTime(df, ctime)
    df[diff_columns] = df[columns].diff(periods=-1).divide(dtime, axis=0)
    return df


def calcVLabel(dflabel, ccx, ccy, ctime):
    """Calculates the velocitiy between centroid points for one label."""
    dtime = calcDTime(dflabel, ctime)
    r = dflabel[[ccx, ccy]].diff(periods=-1)
    v = ((r[ccx] ** 2 + r[ccy] ** 2) ** 0.5) / dtime
    theta = np.arctan2(r[ccy], r[ccx])
    v = np.sign(theta) * v
    return v


def calcVelocity_returnV(df, ccx, ccy, ctime):
    """
    Calculates the velocitiy between centroid points (cols ccx, ccy).
    
    Parameters
    ----------
    df : pd.Dataframe
        Contains tracking data
    ccx: str
       row of  x coordinate of centroid
    ccy: str
        row of y coordinate of centroid
    ctime : str
        row of time

    Returns
    -------
        df : pd.Dataframe
        Contains tracking data with 'v' row
    """
    dfg = df.groupby('label')
    v = dfg.apply(lambda x: calcVLabel(x, ccx, ccy, ctime))
    if v.shape[0] == 1:
        vnew =  v.reset_index().drop('label', axis=1).values.squeeze()
    else:
        vnew = v.reset_index().set_index('level_1').drop('label', axis=1)
    return vnew


def calcVelocity(df, ccx, ccy, ctime):
    df['v'] = calcVelocity_returnV(df, ccx, ccy, ctime)
    return df


def calcSingleFilamentVelocity(df):
    """ Calculates the single filament |v| based on change in abs position."""
    v1 = calcVelocity_returnV(df, 'cx1_ma', 'cy1_ma', 'time')
    v2 = calcVelocity_returnV(df, 'cx2_ma', 'cy2_ma', 'time')
    df['v1'] = v1
    df['v2'] = v2
    return df
