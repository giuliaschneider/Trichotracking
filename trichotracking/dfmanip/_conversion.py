

__all__ = ['convertPxToMeter']


def convertPxToMeter(df, pxCols, umCols, pxLength):
    """ Converts values in columns from pixel to um."""
    df[umCols] = df[pxCols] * pxLength
    return df
