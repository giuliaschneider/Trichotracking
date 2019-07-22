import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ._save import saveplot
from ._constants import *

from IPython.core.debugger import set_trace


def setMeanLine(ax, mean, color, xlim, ypos):
    ax.axvline(mean, color=color, linestyle='dashed', linewidth=1)
    mean_figCoords = (mean - xlim[0]) / (xlim[1] - xlim[0])
    #set_trace()
    """ax.text(mean_figCoords, ypos, '{:.2f}'.format(mean),
             transform=ax.transAxes,horizontalalignment='center',
             size='x-small')"""


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