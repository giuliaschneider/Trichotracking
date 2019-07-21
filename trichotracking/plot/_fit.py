import numpy as np
import pandas as pd


def fit_log(df, xcol, ycol):
    d = df[ (~df[xcol].isnull()) & (~df[ycol].isnull()) ]
    x = d[xcol].values
    y = d[ycol].values
    coeff = np.polyfit(np.log(x), np.log(y), 1)
    return coeff


def fit_semilog(x, y):
    df = pd.DataFrame({'x':x, 'y':y})
    d = df[((~df.x.isin([np.inf, -np.inf, np.nan]))
           &(~np.log(df.y).isin([np.inf, -np.inf, np.nan])))] 
    x = d.x.values
    y = d.y.values
    coeff = np.polyfit((x), np.log(y), 1)
    return coeff
