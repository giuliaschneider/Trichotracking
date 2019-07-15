import os.path
from os.path import join
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from ._constants import *
from ._save import saveplot


from IPython.core.debugger import set_trace


def plot_vts(overlapDir, dflinked, single_tracks, exp, chamber):

    savedir = join(overlapDir, exp, 'v_ts')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    filename = join(savedir, "v_chamber{}.png".format(chamber))


    if 'v_abs' not in dflinked.columns:
        dflinked['v_abs'] = df.v.abs()

    dfg = dflinked[dflinked.label.isin(single_tracks)].groupby('time')
    vmean = dfg.v_abs.mean().values
    t = dfg.frame.first()


    # Set up figure
    figsize = (14,3)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title('Chamber = {}'.format(chamber))
    #dates = mpl.dates.date2num(t.values)
    ax.plot(t, vmean)
    #ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    ax.set_xlabel('Frame')
    ax.set_xlabel('Velocity [µm/s]')
    ax.set_ylim((0,2))
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close('all')


def plot_ts_report(df, labels, filename):
    n = len(labels)
    figsize =(FIGSIZE_TS[0], n*FIGSIZE_TS[1])
    fig, axes = plt.subplots(n, 1, figsize=figsize)

    for ax, label in zip(axes, labels):
        dflabel = df[df.label == label]
        t = (dflabel.time - dflabel.time.iloc[0]).dt.seconds
        y = dflabel.pos_abs_ma
        ax.plot(t, y, color=POS_COLOR)
        ax.set_ylabel(r'$Pos$', color=POS_COLOR)
        y = y[~y.isnull()]
        ylim = (0.8*y.min(), 1.3*y.max())
        ax.set_ylim(ylim)

        ax2 = ax.twinx()
        c2 = cm.tab20b(12)
        ax2.plot(t, dflabel.xlov_norm, 'o', color=LOL_COLOR)
        ax2.plot(t, dflabel.xlov_ma, color=LOL_MA_COLOR)
        ax2.set_ylabel(r'$LOL$', color=LOL_COLOR)
        ax2.set_ylim((-1.25,1.25))

        for tline in t[~np.isnan(dflabel.peaks)]:
            #set_trace()
            ax.axvline(tline, color='gray', lw=0.5)

        ax.tick_params(axis='x', which='both', bottom=False,
                        top=False,labelbottom=False)

    ax.set_xlabel(r'$t$ [s]')
    plt.subplots_adjust(hspace=0)



    fig.savefig(filename, bbox_inches='tight', dpi=300)
    saveplot(fig, filename)
    plt.close('all')
