import numpy as np

__all__ = ['fit_log', 'fit_semilog']


def fit_log(df, xcol, ycol):
    """
    Linearly fits log(x) to log(y) and returns coefficients.

    Parameters
    ----------
    df : pandas Dataframe
    xcol: str
        name of x-column
    ycol: str
        name of y-column

    Returns
    -------
    coeff : tuple
        Coefficients (p0, p1): log(y) = p0*log(x) + p1

    """
    dview = df[[xcol, ycol]]
    xcol_log = xcol + '_log'
    ycol_log = ycol + '_log'
    dview[xcol_log] = np.log(dview[xcol])
    dview[ycol_log] = np.log(dview[ycol])

    coeff = np.polyfit(np.log(dview[xcol].values), np.log(dview[ycol].values), 1)
    return coeff


def fit_semilog(df, xcol, ycol):
    """
    Linearly fits x to log(y) and returns coefficients.

    Parameters
    ----------
    df : pandas Dataframe
    xcol: str
        name of x-column
    ycol: str
        name of y-column

    Returns
    -------
    coeff : tuple
        Coefficients (p0, p1): log(y) = p0*log(x) + p1

    """
    dview = df[[xcol, ycol]]
    ycol_log = ycol + '_log'
    dview[ycol_log] = np.log(dview[ycol])

    coeff = np.polyfit(dview[xcol].values, np.log(dview[ycol].values), 1)
    return coeff
