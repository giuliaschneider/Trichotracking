import numpy as np
import pandas as pd

from IPython.core.debugger import set_trace

def listToColumns(df, col, new_cols):
    """ Splits a column of list to several columns. """
    ncols = len(new_cols)
    dftemp = df[[col]].copy()

    if df[col].isnull().any():
        nrows = dftemp[dftemp[col].isnull()].shape[0]
        nanvalues = (np.empty((nrows, ncols))*(np.nan)).tolist()
        dftemp.loc[dftemp[col].isnull(), col] = nanvalues


    dftemp2 = pd.DataFrame(dftemp[col].values.tolist(), index= dftemp.index)
    df[new_cols] = dftemp2
    return df



def combineNanCols(df, newcol, col1, col2):
    """ Adds newcol to DataFrame which adds values of col2 if col1 is null"""
    df[newcol] = df[col1]
    df.loc[df[newcol].isnull(), newcol] = df.loc[df[newcol].isnull(), col2]
    return df
