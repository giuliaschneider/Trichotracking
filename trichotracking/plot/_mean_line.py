import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from ._save import saveplot
from ._fit import fit_log
from ._constants import *

from IPython.core.debugger import set_trace



def plot_line(ax, df, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
              std, c, label=False):
    groups = df.groupby(pd.cut(df[xcol], bins))
    xmean = groups[xcol].mean()
    ymean = groups[ycol].mean()
    ystd = groups[ycol].std()
    nmean = groups[ycol].count()
    ymean[nmean<4] = np.nan
    ystd[nmean<4] = np.nan
    lower_bound = ymean - ystd
    upper_bound = ymean + ystd
    if label is not False:
        ax.plot(xmean,  ymean, color=c, label=label)
    else:
        ax.plot(xmean,  ymean, color=c)
    if std:
        ax.fill_between(xmean, lower_bound, upper_bound, color=c, alpha=0.5)

def mean_line(df, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
              filename, agg=False, std=True, c=None,
              splitpm=False, loglog=False):

    if c is None:
        c = cm.tab10(0)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)

    if splitpm:
        df1 = df[df[ycol]>0]
        df2 = df[df[ycol]<0]
        plot_line(ax, df1, xcol, ycol, xlabel, ylabel, bins, xlim, ylim, std, c)
        plot_line(ax, df2, xcol, ycol, xlabel, ylabel, bins, xlim, ylim, std, c)
    elif agg:
        df1 = df[df.aggregating]
        df2 = df[~df.aggregating]
        plot_line(ax, df1, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
                  std, AGG_COLOR, label=AGG_LABEL)
        plot_line(ax, df2, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
                  std, NAGG_COLOR, label=NAGG_LABEL)
        plt.legend(frameon=False)

    else:
        plot_line(ax, df, xcol, ycol, xlabel, ylabel, bins, xlim, ylim, std, c)

    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    saveplot(fig, filename)



def plot_allmeanline(df, mdfo, overlapDir):

    xcol = 'xlov_ma'
    ycol = 'vlol'
    ylabel = r'$\bar{v}_{lol}$ [µm/s]'
    xlabel = r'$LOL$'
    ylim = (-1.55,1.55)
    xlim = (-1.1, 1.1)
    bins = np.linspace(-1, 1, 41)

    input = df[(df.v_pos_abs<4)&(df.aggregating)]
    filename = os.path.join(overlapDir, 'lol', 'v_lol_agg')
    mean_line(input, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
                  filename, splitpm=True)
    input = df[(df.v_pos_abs<4)&(~df.aggregating)]
    filename = os.path.join(overlapDir, 'lol', 'v_lol_nagg')
    mean_line(input, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
              filename, c=cm.tab10(1), splitpm=True)


    xcol = 'v_pos_abs'
    ycol = 'hasPeaks'
    xlabel = r'$|\bar{v}_{r}|$ [µm/s]'
    ylabel = r'$\frac{n_{r}}{n_{pairs}}$'
    ylim = (0, 1)
    xlim = (0, 2)
    bins = np.linspace(0, 4, 21)
    print(bins)
    input = mdfo[(mdfo.v_pos_abs<4)&(mdfo.revb!=3)&(mdfo.aggregating)]
    filename = os.path.join(overlapDir, 'v', 'v_nreversing_agg')
    mean_line(input, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
                  filename, std=False)
    input = mdfo[(mdfo.v_pos_abs<4)&(mdfo.revb!=3)]
    filename = os.path.join(overlapDir, 'v', 'v_nreversing_agg_nagg')
    mean_line(input, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
              filename, agg=True, std=False)

    input = mdfo[(mdfo.v_pos_abs<4)&(mdfo.revb!=3)&(~mdfo.aggregating)]
    filename = os.path.join(overlapDir, 'v', 'v_nreversing_nagg')
    mean_line(input, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
              filename, c=cm.tab10(1), std=False)
    ycol = 'peaks'
    ylim = (0, 20)
    input = mdfo[(mdfo.v_pos_abs<4)&(mdfo.revb==2)&(mdfo.aggregating)]
    filename = os.path.join(overlapDir, 'v', 'v_npeaks_agg')
    mean_line(input, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
              filename, c=cm.tab10(1), std=True)
    input = mdfo[(mdfo.v_pos_abs<4)&(mdfo.revb==2)&(~mdfo.aggregating)]
    filename = os.path.join(overlapDir, 'v', 'v_npeaks_nagg')
    mean_line(input, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
              filename, c=cm.tab10(1), std=True)


    mdfo['separates'] = mdfo.breakup == 2
    xcol = 'v_pos_abs'
    ycol = 'separates'
    xlabel = r'$|\bar{v}_{r}|$ [µm/s]'
    ylabel = r'$\frac{n_{separating}}{n_{pairs}}$'
    ylim = (0, 1)
    xlim = (0, 2)
    bins = np.linspace(0, 4, 21)
    print(bins)
    input = mdfo[(mdfo.v_pos_abs<4)&(mdfo.revb!=3)&(mdfo.aggregating)]
    filename = os.path.join(overlapDir, 'v', 'v_separating_agg')
    mean_line(input, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
                  filename, std=False)

    input = mdfo[(mdfo.v_pos_abs<4)&(mdfo.revb!=3)&(~mdfo.aggregating)]
    filename = os.path.join(overlapDir, 'v', 'v_separates_nagg')
    mean_line(input, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
              filename, c=cm.tab10(1), std=False)


    mdfo['tl'] = mdfo.length1 + mdfo.length2

    xcol = 'tl'
    ycol = 'hasPeaks'
    xlabel = r'$l_1 + l_2$ [µm]'
    ylabel = r'$\frac{n_{r}}{n_{pairs}}$'
    ylim = (0, 1)
    xlim = (0, mdfo[xcol].max()*1.1)
    bins = np.linspace(0, mdfo[xcol].max(), 21)
    print(bins)
    input = mdfo[(mdfo.v_pos_abs<4)&(mdfo.revb!=3)&(mdfo.aggregating)]
    filename = os.path.join(overlapDir, 'v', 'l_rev_agg')
    mean_line(input, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
                  filename, std=False)

    input = mdfo[(mdfo.v_pos_abs<4)&(mdfo.revb!=3)&(~mdfo.aggregating)]
    filename = os.path.join(overlapDir, 'v', 'l_rev_nagg')
    mean_line(input, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
              filename, c=cm.tab10(1), std=False)


    slabels = np.unique(df_tracks[df_tracks.type==1].label.values)
    dd = dflinked[dflinked.label.isin(slabels)]

    dfg = dd.groupby('label')
    def msd(cx, cy):
        if cx.size>5:
            x = cx.iloc[0]
            y = cy.iloc[0]
            dist = (cx - x) ** 2  + (cy - y) ** 2
            return dist
        else:
            return cx

    def trun(ct):
        t = ct.iloc[0]
        return pd.to_timedelta(ct-t, unit='s')/ np.timedelta64(1, 's')

    msd2 =  dfg.apply(lambda row:  msd(row.cx, row.cy))
    msd2 = msd2.reset_index().set_index('level_1').drop('label', axis=1)
    tt = dfg.apply(lambda row:  trun(row.time))
    tt = tt.reset_index().set_index('level_1').drop('label', axis=1)
    dd['msd'] = msd2
    dd['tt'] = tt

    xcol = 'tt'
    ycol = 'msd'
    xlabel = r't [s]'
    ylabel = r'MSD [µm]'
    ylim = (1, dd[ycol].max()*1.1)
    xlim = (1, 12000)
    bins = np.linspace(0, 12000, 500)
    print(bins)

    filename = os.path.join(overlapDir, 'v', 'msd')
    mean_line(dd, xcol, ycol, xlabel, ylabel, bins, xlim, ylim,
                  filename, std=False, agg=True, loglog=True)
