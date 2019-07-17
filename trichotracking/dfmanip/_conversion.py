

__all__ = ['convertPxToMeter']



def convertPxToMeter(df, columns, pxLength):
    """ Converts values in columns from pixel to lengths."""
    for col in columns:
        df[col] *= pxLength
    return df


