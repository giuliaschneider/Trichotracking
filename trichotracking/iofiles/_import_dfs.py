import pandas as pd
from dfmanip import combineNanCols, listToColumns

from ._import_df_func import extractValuesFromListOfString as listValues

from IPython.core.debugger import set_trace


def import_dfagg_raw(filepath):
    """ Imports dfagg from text file. """
    # Import dfagg
    dfagg = pd.read_csv(filepath)
    # Convert column of string to column of list
    tracks0 = listValues(dfagg[~dfagg.tracks0.isnull()].tracks0)
    temp =  pd.DataFrame({'tracks0': tracks0},
                         index=dfagg[~dfagg.tracks0.isnull()].index)
    dfagg.loc[~dfagg.tracks0.isnull(), 'tracks0'] = temp
    tracks1 = listValues(dfagg[~dfagg.tracks1.isnull()].tracks1)
    temp =  pd.DataFrame({'tracks1': tracks1},
                         index=dfagg[~dfagg.tracks1.isnull()].index)
    dfagg.loc[~dfagg.tracks1.isnull(), 'tracks1'] = temp

    dfagg.drop_duplicates(subset='trackNr', inplace=True)
    return dfagg


def import_dfagg(filepath):
    """ Imports dfagg from text file and splits tracks into columns. """

    dfagg = import_dfagg_raw(filepath)
    # Split list into single columns
    dfagg = listToColumns(dfagg, 'tracks0', ['tracks00', 'tracks01'])
    dfagg = listToColumns(dfagg, 'tracks1', ['tracks10', 'tracks11'])

    # Create column which has either trackNr before merge or after split
    dfagg = combineNanCols(dfagg, 'deftracks', 'tracks0', 'tracks1')
    # Split column
    dfagg = listToColumns(dfagg, 'deftracks', ['deftrack1', 'deftrack2'])
    aggTracks = dfagg.trackNr.values

    return dfagg


def import_df(filepath):
    df = pd.read_csv(filepath)
    return df
