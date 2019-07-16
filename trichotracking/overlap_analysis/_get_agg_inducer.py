

def getDarkPhases(darkphases, dates):
    """ Return bool array of length dates which is True if it's dark. """
    dark = np.zeros(dates.shape).astype(np.bool)
    if darkphases is not None:
        for dp in darkphases:
            date_light_off = mpl.dates.date2num(dp[0])
            date_light_on = mpl.dates.date2num(dp[1])
            dark = ((dark)
                 | ((dates < date_light_on) & (dates > date_light_off)))
    return dark

def setLightDark(df, darkphases):
    """ Adds a column which is 1 if time is within dark period, else 0."""
    dark = getDarkPhases(darkphases, mpl.dates.date2num(df.time.values))
    df["dark"] = dark
    return df

def get_aggregating(df, agginducer, aggchamber):
    """ Return bool array, True if aggregation is induced. """
    if agginducer == 'light':
        aggregating = ~getDarkPhases(darkphases,
                                     mpl.dates.date2num(df.time.values))
    elif agginducer == 'chamber':
        aggregating= df.apply(lambda row: row["label"][2] in aggchamber,
                              axis=1)
    else:
        aggregating = np.zeros(df.shape[0]).astype(np.bool)
    return aggregating
