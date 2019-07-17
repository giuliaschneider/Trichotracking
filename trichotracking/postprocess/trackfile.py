import numpy as np
import pandas as pd

from dfmanip import (calcMovingAverages,
                     calcPeaksSingle,
                     calcVelocity,
                     convertPxToMeter,)


def trackfile(trackFile, timesFile, pxConversion):
    """
    Imports raw track data and calculates velocity and peaks.
    
    Parameters
    ----------
    trackFile : str
        FIle path to track data
    timesFile : str
        FIle path to track data
    ccy: str
        row of y coordinate of centroid
    ctime : str
        row of time

    Returns
    -------
        df : pd.Dataframe
        Contains tracking data with 'v' row
    """


    cols = ["trackNr", "length", "cx","cy"]
    df = pd.read_csv(trackFile, usecols=cols)


    columns = ["length", "cx","cy"]
    df = convertPxToMeter(df, columns, pxConversion)
    df['label'] = df.trackNr.astype('int')
    columns = ["cx", "cy"]
    ma_columns =  ["cx_ma","cy_ma"]
    df = calcMovingAverages(df, 13, columns, ma_columns)
    times = np.loadtxt(timesFile)
    df['time'] = times[df.frame]
    #df.time = pd.to_datetime(df.time, format='%Y-%m-%d %H:%M:%S.%f')

    df = calcVelocity(df, 'cx_ma', 'cy_ma', 'time')
    df['v_abs'] = df.v.abs()

    df = calcPeaksSingle(df, 'v')
    return df