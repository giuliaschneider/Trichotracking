import numpy as np
import pandas as pd
from os.path import join
from datetime import datetime

from dfmeta import  (get_agginducer,
                     get_aggchambers,
                     get_chambers,
                     get_dirs,
                     get_darkphases,
                     get_files,
                     get_px)
from dfmanip import (calcChangeInCol,
                     calcLongVelocity,
                     calcMovingAverages,
                     calcLabel,
                     calcPeaks,
                     calcPeaksSingle,
                     convertPxToMeter)


from ._get_agg_inducer import get_aggregating



def import_dflinked(dffile, timeFile, exp, chamber, agginducer, aggchambers,
                     pxLength):

    cols = pd.read_csv(dffile, index_col=0, nrows=0).columns
    cols = cols[~cols.str.startswith('Unnamed')]
    cols = cols[~cols.isin(['contours', 'pixellist_xcoord', 'pixellist_ycoord'])]
    # dflinked.drop(columns=)
    df = pd.read_csv(dffile, usecols=cols, index_col=None, header=0)
    time_chamber =  np.loadtxt(timeFile)
    df["exp"] = exp
    trackNrs = df.trackNr.values
    df["label"] = calcLabel(trackNrs, chamber, exp)
    df['aggregating'] = get_aggregating(df, agginducer, aggchambers)
    # Add time column
    time = time_chamber[df.frame]
    time_dt = [datetime.fromtimestamp(t) for t in time]
    df['time'] = time_dt
    return df



def import_alldflinked(dfMeta, expLabels=None):
    df_list = []
    if expLabels is None:
        expLabels = np.unique(dfMeta.exp)

    # Iterate through all experiments
    for exp in expLabels:
        dataDir, dataDirs, resultDir, resultDirs = get_dirs(dfMeta, exp)
        trackFiles, aggFiles, dftrackFiles, timesFiles = get_files(dfMeta, exp)
        chambers = get_chambers(dfMeta, exp)
        agginducer = get_agginducer(dfMeta, exp)
        aggchambers = get_aggchambers(dfMeta, exp)
        pxLength = get_px(dfMeta, exp)

        for trackFile, timeFile, chamber in zip(trackFiles, timesFiles, chambers):
            df_c = import_dflinked(trackFile, timeFile, exp, chamber,
                                   agginducer, aggchambers, pxLength)
            df_list.append(df_c)

        print("Experiment " + exp + ": Imported dflinked")


    df = pd.concat(df_list, axis=0, ignore_index=True, sort=False)
    df.sort_values(by=['label','frame'], inplace=True)


    df.time = pd.to_datetime(df.time, format='%Y-%m-%d %H:%M:%S.%f')
    # Convert from pixel to length
    cols = ['length', 'cx', 'cy']
    df = convertPxToMeter(df, cols, pxLength)
    # Moving average
    cols = ['length']
    ma_cols =['length_ma']
    df = calcMovingAverages(df, 5, cols, ma_cols)
    cols = ['cx', 'cy']
    ma_cols =['cx_ma', 'cy_ma']
    df = calcMovingAverages(df, 7, cols, ma_cols)
    print("Calculated Moving Average")


    # Calculate velocity
    df = calcLongVelocity(df, 'cx_ma', 'cy_ma', 'time')
    df['v_abs'] = df.v.abs()
    columns = ["v"]
    ma_columns =  ["v_ma"]
    df = calcMovingAverages(df, 5, columns, ma_columns)
    print("Calculated velocity")

    df = calcPeaksSingle(df, 'v_ma')


    # Calculalte ratio of eigenvalues
    df['ews'] = df['ew2']/df['ew1']
    return df
