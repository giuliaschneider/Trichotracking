from os.path import abspath, join
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from overlap_analysis import average_df

mpl.rcParams.update({'font.size': 12})


def plot_scatter_peaks(df_peaks):
    # Define colors
    df_peaks['color'] = df_peaks.groupby('label').ngroup()
    df_peaks['color'] = df_peaks['color'] / df_peaks.label.nunique()
    fig = plt.figure()
    plt.scatter(df_peaks.length1-df_peaks.length2, df_peaks.xlov_ma.abs(),
                c=df_peaks.color, cmap='jet')
    plt.xlabel(r'$l_{short}/l_{long}$')
    plt.ylabel(r'$|loo_{reversal}|$')
    plt.title('Light switching experiment Uli & Giulia')
    #fig.savefig(join(saveDirHist, "02_loo_tracks"))


def plot_errorplot(dfg_agg, col, dfMeta, filename, ylabel, ylim=None):
    """ Plots an errorplot, values divided on experiment and dark/light."""
    # Set up figure
    fig= plt.figure()
    ax = plt.subplot(111)

    # Indexes =
    ind = np.arange(dfg_agg.shape[0])

    # Iterate through all experiments
    expLabels = np.unique(dfg_agg.index.get_level_values('exp'))
    aInd = 0
    for exp in expLabels:
        try:
            y = dfg_agg.loc[dfg_agg.index.get_level_values('exp')==exp,
                           col]['mean']
        except: set_trace()
        err = dfg_agg.loc[dfg_agg.index.get_level_values('exp')==exp,
                       col]['std']
        bInd = aInd + y.size
        plt.errorbar(ind[aInd:bInd], y, err, linestyle='None', marker='o',
                     label=dfMeta[dfMeta.exp==exp].title.values[0])
        aInd = bInd

    # Set figure properties
    ax.set_xticks(ind)
    ax.set_xticklabels(dfg_agg['plotLabel'])
    if ylim is not None: ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    plt.legend()
    fig.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_errorplots(df, dfMeta, saveDir, filename, ylabel, col, agg='mean',
    colNorm=None, aggNorm='mean',ylim=None):
    """ Plots the errorplot for values in df.col agg for each experiment."""
    # Aggregate dataframe
    dfg_agg = average_df(df, dfMeta, col, agg, colNorm, aggNorm)

    # Plot not normed value
    fname = join(saveDir, filename)
    plot_errorplot(dfg_agg, col, dfMeta, fname, ylabel, ylim=ylim)

    # Plot not normed value
    #fname = join(saveDir, filename+'_norm')
    #plot_errorplot(dfg_agg, 'norm', dfMeta, fname, ylabel, ylim=ylim)
