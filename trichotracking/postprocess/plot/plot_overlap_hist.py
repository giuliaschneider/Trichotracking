import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms
from matplotlib.collections import LineCollection
from matplotlib import cm
from overlap.analysis_overlap import *

from hist_functions import fit_kde, get_mean, fit_bigaussian

mpl.rcParams.update({'font.size': 12})




def plotHistNQuantities(listOfS, labels, title, xlabel, xlims, nbins,
                        filename, legT='', bandwidth=0.08):

    # Sets histogram parameters
    if xlims is None:
        if s2 is None:
            xlims = (s1.min(), s1.max())
        else:
            xlims = (min(s1.min(), s2.min()), max(s1.max(), s2.max()))
    if (nbins is None):
        nbins = 'auto'
    props = {'bins':'auto', 'density':True}

    # Plot histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = [cm.tab10(i) for i in range(len(labels))]

    for s, label, c in zip(listOfS, labels, colors):
        if len(s)<1 or s is None:
            label = label + ", n = {}".format(0)
            plt.plot([], color=c, label=label)
        else:
            n = s.size
            label = label + ", n = {}".format(n)
            xplot, freq, kde = fit_kde(s, xlims, bandwidth)
            ax.hist(s[~np.isnan(s)].values, alpha=0.5, color=c, **props)
            ax.plot(xplot, freq, color=c, label=label)
            s1mean = get_mean(s, kde, xlims)
            setMeanLine(ax, s1mean, c, xlims, 1.06)

    plt.xlim(xlims)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend(frameon=False, title=legT)
    fig.savefig(filename, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)


def plotHistTwoQuantities(s1, s2, label1, label2, title,
                          xlabel, xlims, nbins, filename, legT='',
                          bandwidth=0.08):
    """ Plots the histogram of the two arrays s1, s2. """

    plotHistNQuantities([s1, s2], [label1, label2], title, xlabel,
                        xlims, nbins, filename, legT, bandwidth)

def plotHistExperiments(df, col, dfMeta, saveDir, filename, labels1,
    condition1, labels2, condition2, xlabel, xlims=None, nbins=None):
    """ Plots the the histogram for each experiment. """

    expLabels = dfMeta.exp
    titles = dfMeta.title

    # Iterate all experimetns
    for i, exp in enumerate(expLabels):
        s1 = df[(condition1) & (df.exp==exp)][col]
        if condition2 is None:
            s2 = None
        else:
            s2 = df[(condition2) & (df.exp==exp)][col]
        title = titles[i]

        if filename.endswith('.png'):
            fname_exp = os.path.join(saveDir, filename)
        else:
            fname_exp = os.path.join(saveDir, filename+"_"+exp+'.png')
        plotHistTwoQuantities(s1, s2, labels1[i], labels2[i], title,
                              xlabel, xlims, nbins, fname_exp)


def plotHist(df, col, condition1, condition2, label1, label2, saveDir,
             filename, title, xlabel, xlims=None, nbins=None):

    s1 = df[condition1][col]
    if condition2 is None:
        s2 = None
    else:
        s2 = df[condition2][col]
    filename = os.path.join(saveDir, filename)
    plotHist(s1, s2, label1, label2, title, xlabel, xlims, nbins,
             filename)
