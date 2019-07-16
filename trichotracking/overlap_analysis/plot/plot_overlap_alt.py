




def plotHistXlovAll(df, col, nbins, saveDir, filename, title, label):
    bins = np.linspace(0,1,nbins)
    # All Data
    df_all = df[(df.xlov_ma.abs()>0.01)]
    filename = os.path.join(saveDir, filename)


    # Peak data
    df_peak = df[(df.peaks==1) & (df.xlov_ma.abs()>0.01)]
    filename = os.path.join(saveDir, filename + "_peaks")
    bins = np.linspace(0,1,12)
    label = label + " peaks"
    title = title + " peaks"
    plotHist(bins, df_peak[col].abs(), None,
             label, None, title, filename)
    plt.close('all')

def plotHistXlov(df, col, nbins, saveDir, filename, title,
                 labelDark, labelLight):
    bins = np.linspace(0,1,nbins)
    # All Data
    df_dark = df[(df.dark==1) & (df.xlov_ma.abs()>0.01)]
    df_light = df[(df.dark==0) & (df.xlov_ma.abs()>0.01)]
    filename = os.path.join(saveDir, filename)
    plotHist(bins, df_dark[col].abs(), df_light[col].abs(),
             labelDark, labelLight, title, filename)

    # Peak data
    df_dark = df[(df.dark==1) & (df.peaks==1) & (df.xlov_ma.abs()>0.01)]
    df_light = df[(df.dark==0) & (df.peaks==1) & (df.xlov_ma.abs()>0.01)]
    filename = os.path.join(saveDir, filename + "_peaks")
    bins = np.linspace(0,1,12)
    plotHist(bins, df_dark[col].abs(), df_light[col].abs(),
             labelDark, labelLight, title, filename)
    plt.close('all')


def plotHistPos(df, col, nbins, saveDir, filename, labelDark, labelLight):

    bins = np.linspace(-1,2,nbins)

    # All Data
    df_dark = df[(df.dark==1)]
    df_light = df[(df.dark==0)]
    filename = os.path.join(saveDir, filename)
    title = filename[:-5]
    plotHist(bins, df_dark[col].abs(), df_light[col].abs(),
             labelDark, labelLight, title, filename)

    # Peak data
    df_dark = df[(df.dark==1) & (df.peaks==1)]
    df_light = df[(df.dark==0) & (df.peaks==1)]
    filename = os.path.join(saveDir, filename + "_peaks")

    plotHist(bins, df_dark[col].abs(), df_light[col].abs(),
             labelDark, labelLight, title, filename)

    plt.close('all')


def plotHistXlovTracks(df,  nbins, saveDir, labelDark, labelLight):
    trackLabels = np.unique(df.label)

    for l in trackLabels:
        # All Data
        df_track = df[(df.label == l) & (df.xlov_ma != 0)]
        df_dark = df_track[(df_track.dark==1)]
        df_light = df_track[(df_track.dark==0)]
        track = int(df_track.track.values[0])
        title = "Track = {}".format(track)
        label = df_track.label.values[0]
        filename = os.path.join(saveDir, "{}_xlov".format(label))
        bins = np.linspace(0,1,nbins)
        plotHist(bins, df_dark.xlov_ma.abs(), df_light.xlov_ma,
                 labelDark, labelLight, title, filename)

        # Peak data
        df_dark = df[(df.dark==1) & (df.peaks==1)]
        df_light = df[(df.dark==0) & (df.peaks==1)]
        track = int(df_track.track.values[0])
        title = "Track = {}".format(track)
        label = df_track.label.values[0]
        filename = os.path.join(saveDir, "{}_xlov_peaks".format(label))
        plotHist(bins, df_dark.xlov_ma, df_light.xlov_ma,
                 labelDark, labelLight, title, filename)

    plt.close('all')
