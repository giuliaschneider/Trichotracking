

def filter_dfg(dfg, trackcol, col, minValue, maxValue):
    """ Returns trackNr larger than minValue & smaller than maxValue."""
    tracks = dfg[((dfg[col] > minValue)
                 &(dfg[col] < maxValue))][trackcol]
    return tracks.values
