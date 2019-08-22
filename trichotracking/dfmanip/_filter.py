import numpy as np

__all__ = ['removeNan']


def removeNan(df, cols):
    """ Returns dataframe view, all NaN-rows in cols removed.

    Parameters
    ----------
    df : pandas Dataframe
    cols: list of str
        columns which are filtered for NaNs 

    Returns
    -------
    d : pandas Dataframe
        Filtered dataframe

    """
    if (isinstance(cols, 'str')):
        cols = [cols]

    d = df
    for col in cols:
        d = d[~d[col].isin([np.inf, -np.inf, np.nan])]
    return d
