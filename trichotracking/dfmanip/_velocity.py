import numpy as np
import numpy.linalg as la
import pandas as pd


__all__  = ['calcChangeInTime', 'calcVelocity', 'calcSingleFilamentVelocity'] 



def calcDTime(df, ctime):
    time = df[ctime].diff(periods=-1).values
    dtime = pd.to_timedelta(time, unit='s') / np.timedelta64(1, 's')
    return dtime


def calcChangeInTime(df, ctime, columns, diff_columns):
    """ Calculates the change in overlap for all tracks. """
    dtime = calcDTime(df, ctime)
    df[diff_columns] = df[columns].diff(periods=-1).divide(dtime, axis=0)
    return df


def calcVLabel(dflabel, ccx, ccy, ctime):
    """Calculates the velocitiy between centroid points for one label."""
    dtime = calcDTime(dflabel, ctime)
    r = dflabel[[ccx, ccy]].diff(periods=-1)
    v = ((r[ccx]**2 + r[ccy]**2)**0.5) / dtime
    theta = np.arctan2(r[ccy], r[ccx])
    v = np.sign(theta)*v
    return v

def calcVelocity(df, ccx, ccy, ctime):
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
    vnew = v.reset_index().set_index('level_1').drop('label', axis=1)
    df['v'] = vnew
    return df


def calcSingleFilamentVelocity(df):
    """ Calculates the single filament v based on change in abs position."""
    time = df["time"].diff(periods=-1).values
    dtime = pd.to_timedelta(time, unit='s') / np.timedelta64(1, 's')
    labels = np.unique(df.label)

    n_cols = ["v1", "v2"]
    for col in n_cols:
        df[col] = np.nan
    for label in labels:
        time = df[df.label==label].time.diff(periods=-1).values
        dtime = pd.to_timedelta(time, unit='s') / np.timedelta64(1, 's')
        df_label = df[df.label==label]
        # Direction vector of long filament
        long_fil_dir = df_label[["dirx1","diry1"]]
        # Movement vector of long/short filament
        r1 = df_label[["cx1_ma","cy1_ma"]].diff(periods=-1).values
        r2 = df_label[["cx2_ma","cy2_ma"]].diff(periods=-1).values
        sign1 = np.sign((r1 * long_fil_dir).sum(axis=1))
        sign2 = np.sign((r2 * long_fil_dir).sum(axis=1))
        # Signed velocities
        df.loc[df.label==label, "v1"] = sign1 * la.norm(r1,axis=1)/dtime
        df.loc[df.label==label, "v2"] = sign2 * la.norm(r2,axis=1)/dtime

    return df
