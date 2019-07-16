import numpy as np
import pandas as pd

from dfmeta import  (get_agginducer,
                     get_aggchambers,
                     get_chambers,
                     get_dirs,
                     get_darkphases,
                     get_px)
from dfmanip import convertPxToMeter
from iofiles import find_txt

from ._get_agg_inducer import get_aggregating

def import_dfoverlap(filepath, exp, chamber, agginducer, aggchambers,
                     pxLength):
    df = pd.read_csv(filepath, index_col=None, header=0)
    df.time = pd.to_datetime(df.time, format='%Y-%m-%d %H:%M:%S.%f')
    trackNr = int(df.track.values[0])
    df["exp"] = exp
    df["label"] = exp + str(chamber) + "{:04}".format(trackNr)
    df['aggregating'] = get_aggregating(df, agginducer, aggchambers)

    # Convert from pixel units to length units
    columns = ["length1", "length2", "length_overlap", "xlov",
               "ylov", "cx1", "cx2", "cy1", "cy2"]
    df = convertPxToMeter(df, columns, pxLength)
    return df

def import_all_dfoverlap(dfMeta, expLabels=None):
    """ Importes all dataframes given in listOfFiles"""
    # Initalize list of dataframes
    df_list = []
    if expLabels is None:
        expLabels = np.unique(dfMeta.exp)

    # Iterate through all experiments
    for exp in expLabels:
        dataDir, dataDirs, resultDir, resultDirs = get_dirs(dfMeta, exp)
        chambers = get_chambers(dfMeta, exp)
        agginducer = get_agginducer(dfMeta, exp)
        aggchambers = get_aggchambers(dfMeta, exp)
        pxLength = get_px(dfMeta, exp)

        for dir, chamber in zip(resultDirs, chambers):
            trackFiles = find_txt(dir, 'track')

            # Iterate through tracks
            for file in trackFiles:
                track = import_dfoverlap(file, exp, chamber, agginducer,
                                         aggchambers, pxLength)
                df_list.append(track)

        print("Experiment " + exp + ": Imported dfoverlap")

    df = pd.concat(df_list, axis=0, ignore_index=True)
    df.time = pd.to_datetime(df.time, format='%Y-%m-%d %H:%M:%S.%f')
    df.sort_values(by=['label', 'time'], inplace=True)
    df['index'] = np.arange(df.shape[0])
    df.set_index('index', inplace=True)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    return df
