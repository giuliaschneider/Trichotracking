import numpy as np

__all__ = ['calcMovingAverages']


def calcMovingAverages(df, wsize, columns, ma_columns):
    """ Calculates the moving average for all given columns. """
    for col in ma_columns:
        df[col] = np.nan
    dfg = df.groupby('label')

    df[ma_columns] = dfg.apply(lambda x: x[columns].rolling(window=wsize, center=True).mean())
    return df
