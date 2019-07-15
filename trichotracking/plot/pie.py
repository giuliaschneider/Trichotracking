import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp

from ._save import saveplot
from ._fit import fit_semilog
from ._constants import *

from IPython.core.debugger import set_trace


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n(n={:d})".format(pct, absolute)


def pie_chart(data, labels, colors, filename):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Plot

    wedges, texts, autotexts = ax.pie(data, labels=labels,colors=colors,
                                      autopct='%1.1f%%',
                                      textprops=dict(color="k"))

    textstr = r'$n=$%d'%np.sum(data)
    pos = (0.5, 0.5)
    ax.annotate(textstr, xy=pos, xytext=pos, xycoords='figure fraction',
                ha='center', va='center', color="k", bbox=dict(edgecolor='k', facecolor=(0.1, 0.2, 0.5, 0.0)))


    plt.axis('equal')
    saveplot(fig, filename)
    plt.close('all')


def bars_of_breakup(mdfo, overlapDir):
    savedir = os.path.join(overlapDir, 'breakup')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    mycmap = cm.Set2((0,2,4,5,6,7))
    mycmap = mpl.colors.ListedColormap(mycmap, name='mycmap')

    mdfo.breakup.fillna(4, inplace=True)

    all = mdfo.groupby('breakup').breakup.count()[[1,2,3,5,4]].values
    agg = mdfo[mdfo.aggregating].groupby('breakup').breakup.count()[[1,2,3,5,4]].values
    nagg = mdfo[~mdfo.aggregating].groupby('breakup').breakup.count()[[1,2,3,5,4]].values
    rev = mdfo[mdfo.hasPeaks].groupby('breakup').breakup.count()[[1,2,3,5,4]].values
    nrev =mdfo[~mdfo.hasPeaks].groupby('breakup').breakup.count()[[1,2,3,5,4]].values

    index = ['Merged with other filament(s)', 'Split', 'Experiment ended', 'Left field of view', 'Unknown']

    data = pd.DataFrame({'Aggregating': agg/agg.sum(), 'Non-aggregating': nagg/nagg.sum(),
                         #'Reversing': rev/rev.sum(), 'Non-reversing': nrev/nrev.sum(),
                         'Track End': index})
    data.set_index(data['Track End'], inplace=True)

    ax = data.T[:2].plot(kind='bar', stacked=True, cmap=mycmap)
    plt.gcf().set_size_inches((12/2.54, 4.5/2.54))
    ax.set_ylim((0, 1))
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend = ax.get_legend()
    legend.get_frame().set_edgecolor('none')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    bb = legend.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    # Change to location of the legend.
    bb.x0 = 0.97
    bb.y0 = 0.65
    bb.y1 = 0.85
    legend.set_bbox_to_anchor(bb,)
    plt.subplots_adjust(bottom=0.2, right=0.8, top=0.9)

    ns = np.array([agg.sum(), nagg.sum(), rev.sum(), nrev.sum()])
    for i, cond in enumerate([(mdfo.aggregating), (~mdfo.aggregating)]):
        textstr = r'$n=$%d'%ns[i]
        pos = (i, 1.05)
        ax.annotate(textstr, xy=pos, xytext=pos, annotation_clip=False,
                    ha='center', va='center', color="k")

        j = 0.05
        for t in [1, 2]:
            pos = (i, j)
            fr = mdfo[cond & (mdfo.breakup==t)].breakup.count() / ns[i]
            textstr = '{:.2f}'.format(fr)
            ax.annotate(textstr, xy=pos, xytext=pos, annotation_clip=False,
                        ha='center', va='center', color="w")

            j += fr

    #legend.set_loc('lower center')
    legend.set_title('')
    filename = os.path.join(savedir, 'breakup.png')
    saveplot(plt.gcf(), filename)
    plt.close('all')


def bars_of_agg(mdfo, overlapDir):
    savedir = os.path.join(overlapDir, 'breakup')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    agg = mdfo[mdfo.aggregating].groupby('revb').revb.count().values
    nagg = mdfo[~mdfo.aggregating].groupby('revb').revb.count().values
    index = REV_SPLIT_LABEL

    data = pd.DataFrame({'Aggregating': agg/agg.sum(), 'Non-aggregating': nagg/nagg.sum(), 'Group': index})
    data.set_index(data['Group'], inplace=True)

    mycmap = cm.tab20((12,13,14,15))
    mycmap = mpl.colors.ListedColormap(mycmap, name='mycmap')

    ax = data.T[:2].plot(kind='bar', stacked=True, cmap=mycmap)
    plt.gcf().set_size_inches((12/2.54, 4.5/2.54))
    ax.set_ylim((0, 1))
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend = ax.get_legend()
    legend.get_frame().set_edgecolor('none')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    bb = legend.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    # Change to location of the legend.
    bb.x0 = 0.97
    bb.y0 = 0.4
    bb.y1 = 0.6
    legend.set_bbox_to_anchor(bb,)
    plt.subplots_adjust(bottom=0.2, right=0.8, top=0.9)

    ns = np.array([agg.sum(), nagg.sum()])
    for i, cond in enumerate([(mdfo.aggregating), (~mdfo.aggregating)]):
        textstr = r'$n=$%d'%ns[i]
        pos = (i, 1.05)
        ax.annotate(textstr, xy=pos, xytext=pos, annotation_clip=False,
                    ha='center', va='center', color="k")

        j = 0.05
        for t in range(1,5):
            pos = (i, j)
            fr = mdfo[cond & (mdfo.revb==t)].revb.count() / ns[i]
            textstr = '{:.2f}'.format(fr)
            ax.annotate(textstr, xy=pos, xytext=pos, annotation_clip=False,
                        ha='center', va='center', color="w")

            j += fr

    #legend.set_loc('lower center')
    legend.set_title('')
    filename = os.path.join(savedir, 'breakup_reversing.png')
    saveplot(plt.gcf(), filename)
    plt.close('all')
