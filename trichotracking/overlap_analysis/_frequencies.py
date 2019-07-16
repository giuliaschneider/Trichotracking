

def calcFrequencies(df_track, dt):
    """ Calculates the frequency spectrum of time series in df_track."""
    tmin = df_track.time.min()
    tmax = df_track.time.max()
    df_track.set_index('time', inplace=True)
    regular_times =pd.date_range(tmin, tmax, freq='{}s'.format(dt))
    pos_original = df_track.pos_ma
    pos_regular = pd.Series(np.nan, index=regular_times)
    pos_regular = pd.concat([pos_original, pos_regular])
    pos_regular = pos_regular[~pos_regular.index.duplicated(keep='first')]
    pos_regular.interpolate(method='time', inplace=True)
    pos_regular= pos_regular[regular_times]
    p = pos_regular.values
    p = p[~np.isnan(p)]
    f = fft.fft(p)
    f = fft.fftshift(f)
    freq = fft.fftfreq(p.shape[0], dt)
    freq = fft.fftshift(freq)
    return f, freq


def calcReversalPeriod(df, dt):
    """ Calculates the reversal period for each track."""
    labels = np.unique(df.label)
    df['Treverse'] = np.nan
    for label in labels:
        df_track = df[df.label==label]
        f, freq = calcFrequencies(df_track, dt)
        fabs = np.abs(f)
        I = np.argsort(fabs)
        fmajor = np.abs(freq[I][-2])
        dftmajor = 1.1*fabs[I][-2]
        df.loc[df.label==label, 'Treverse'] = 1/fmajor/60
    return df
