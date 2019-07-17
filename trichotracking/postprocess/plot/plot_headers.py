from os.path import join
from overlap.plot_overlap_hist import *
from overlap.plot_overlap_phase import *


def plotHistVSingle(df, dfMeta, saveDir):
    """ Plots the LOL histogram for each experiment. """
    col = 'v_abs'
    filename = 'hist_v'
    labels1 = dfMeta.labelDark
    cond1 = ((df.dark==1))
    labels2 = dfMeta.labelLight
    cond2 = (df.dark==0)
    xlabel = r'$v_{ma}$'

    plotHistExperiments(df, col, dfMeta, saveDir, filename, labels1,
        cond1, labels2, cond2, xlabel, xlims=(0,2.25), nbins=None)


def plotHistVFil(df, dfMeta, saveDir):
    """ Plots the LOL histogram for each experiment. """
    expLabels = np.unique(dfMeta.exp)
    cols = ['v1_abs', 'v2_abs']
    filenames = ['v1', 'v2']
    xlabel = r'$v_{Filament}$'

    # Iterate all experimetns
    for exp in expLabels:
        trackLabels = np.unique(df[df.exp==exp].label)
        labels1 = dfMeta[dfMeta.exp==exp].labelDark.values
        labels2 = dfMeta[dfMeta.exp==exp].labelLight.values

        # Iterate all labels
        for l in trackLabels:
            df_track = df[df.label == l]
            cond1 = ((df_track.dark==1))
            cond2 = (df_track.dark==0)

            # Define title
            track = int(df_track.track.values[0])
            dl = df_track.l2_ma.mean() / df_track.l1_ma.mean()
            title = "Track = {}".format(track)

            for col, fname in zip(cols, filenames):

                # Define filename
                label = df_track.label.values[0]
                saveDirexp = os.path.join(saveDir, exp)
                filename = "{}_{}.png".format(label, fname)

                plotHistExperiments(df_track, col, dfMeta, saveDir, filename,
                    labels1, cond1, labels2, cond2, xlabel, xlims=(0,2.25),
                    nbins=None)


def plotHistLOL(df, dfMeta, saveDir):
    """ Plots the LOL histogram for each experiment. """
    col = 'xlov_ma_abs'
    filename = 'hist_lol',
    labels1 = dfMeta.labelDark
    cond1 = ((df.peaks==1) & (df.xlov_ma.abs()>0.01) & (df.dark==1))
    labels2 = dfMeta.labelLight,
    cond2 = ((df.peaks==1) & (df.xlov_ma.abs()>0.01) & (df.dark==0))
    xlabel = r'$LOL_{ma}$'

    plotHistExperiments(df, col, dfMetaExp, saveDir, filename, labels1,
        cond1, labels2, cond2, xlabel, xlims=(0,1), nbins=None)


def plotErrorLOL(df, dfMeta, saveDir):
    ylabel = r'$LOL_{reversal}/v_{pos}$'
    filename = 'errorplot_lol'
    plot_errorplots(df, dfMetaExp, saveDir, filename, ylabel,'xlov_ma_abs_peaks',
        agg='mean', colNorm='v_pos_abs', aggNorm='mean', ylim=None)

def plotLOLvsVAll(df, dfMeta, saveDir):
    dfg = df.groupby(['label']).mean()

    fig = plt.figure()
    plt.plot(dfg.xlov_ma_abs_peaks, dfg.v_pos_abs, 'o')
    plt.xlabel(r'$LOL_{reversal}$')
    plt.ylabel(r'$v_{pos}$')
    filename = join(saveDir, "lol_v_all")
    fig.savefig(filename, bbox_inches='tight', dpi=150)



def plotLOLvsV(df, dfMeta, saveDir):
    dfg = df.groupby(['exp', 'label']).mean()
    expLabels = np.unique(dfg.index.get_level_values('exp'))
    for exp in expLabels:
        fig = plt.figure()
        x = dfg.loc[dfg.index.get_level_values('exp')==exp,'xlov_ma_abs_peaks']
        y = dfg.loc[dfg.index.get_level_values('exp')==exp,'v_pos_abs']
        plt.plot(x, y, 'o')
        plt.xlabel(r'$LOL_{reversal}$')
        plt.ylabel(r'$v_{pos}$')
        plt.xlim((0,1))
        plt.ylim((0,2.25))
        filename = join(saveDir, "lol_v_exp{}".format(exp))
        fig.savefig(filename, bbox_inches='tight', dpi=150)

        if exp=='005':
            print(dfg.loc[(dfg.index.get_level_values('exp')==exp),
                        ['xlov_ma_abs_peaks', 'v_pos_abs']])

def plotLOLvsV_darklight(df, dfMeta, saveDir):
    dfg = df.groupby(['exp', 'label', 'dark']).mean()
    expLabels = np.unique(dfg.index.get_level_values('exp'))
    for exp in expLabels:
        fig = plt.figure()
        x = dfg.loc[(dfg.index.get_level_values('exp')==exp)
                    &(dfg.index.get_level_values('dark')==1),
                    'xlov_ma_abs_peaks']
        y = dfg.loc[(dfg.index.get_level_values('exp')==exp)
                    &(dfg.index.get_level_values('dark')==1),
                    'v_pos_abs']
        plt.plot(x, y, 'o', label=dfMeta[dfMeta.exp==exp].labelDark.values[0])
        x = dfg.loc[(dfg.index.get_level_values('exp')==exp)
                    &(dfg.index.get_level_values('dark')==0),
                    'xlov_ma_abs_peaks']
        y = dfg.loc[(dfg.index.get_level_values('exp')==exp)
                    &(dfg.index.get_level_values('dark')==0),
                    'v_pos_abs']
        plt.plot(x, y, 'o', label=dfMeta[dfMeta.exp==exp].labelLight.values[0])
        plt.xlabel(r'$LOL_{reversal}$')
        plt.ylabel(r'$v_{pos}$')
        plt.xlim((0,1))
        plt.ylim((0,2.25))

        plt.legend(frameon=False,)
        filename = join(saveDir, "lol_v_exp{}_ld".format(exp))
        fig.savefig(filename, bbox_inches='tight', dpi=150)

        if exp=='005':
            print(dfg.loc[(dfg.index.get_level_values('exp')==exp),
                        ['xlov_ma_abs_peaks', 'v_pos_abs']])
