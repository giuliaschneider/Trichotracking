import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms
from matplotlib.collections import LineCollection
from matplotlib import cm


def plotTracksLOLAll(df, dfMeta, saveDir):
    """ Plots the LOL time series for all tracks in df."""
    # Time series settings
    props_marker = {'ls':'', 'marker': 'o', 'ms':4}
    props_line = {'ls':'-', 'marker':''}
    xlabel = ['Time [hh:mm]']

    col = ("xlov_norm", "xlov_ma",)
    label = (r"$LOL$", r"$LOL_{ma}$",)
    prop = (props_marker, props_line,)
    ylabel = r'$LOL$'
    ylim = (-1.1, 1.1)
    hlines = [0,]
    color = (cm.tab20(8), cm.tab20(9))

    trackLabels = np.unique(df.label).tolist()
    temp = trackLabels[0]
    trackLabels.pop(0)
    trackLabels.append(temp)

    # Set up figure
    mpl.rcParams.update({'font.size': 12})
    nAxis = len(trackLabels)
    figsize = (14,6)
    fig, axes = plt.subplots(nAxis, 1, sharex=True, figsize=figsize)

    # Iterate all labels in dataframe
    for i, track in enumerate(trackLabels):
        ax = axes[i]
        if i < nAxis-1:
            plt.setp(ax.get_xticklabels(), visible=False)

        df_track = df[df.label == track]
        exp = df_track.exp.values[0]

        dates = mpl.dates.date2num(df_track.time.values)


        # Get darkphases
        darkphases = dfMeta[dfMeta.exp==exp].darkphases.values[0]
        if darkphases is not None:
            dark = getDarkPhases(darkphases, dates)

        # Norm dates
        minDate = df_track.time.min()
        startDate = datetime(1970,1,1)
        #set_trace()
        dates = mpl.dates.date2num((startDate+(df_track.time - minDate)).values)

        # Shade dark areas
        if darkphases is not None:
            trans = mtransforms.blended_transform_factory(ax.transData,
                                                          ax.transAxes)
            ax.fill_between(dates, 0, 1, where=(dark),transform=trans,
                             facecolor=cm.tab20(15), alpha=0.7)


        # Plot data
        for c, p, l, cc in zip(col, prop, label, color):
            ax.plot_date(dates, df_track[c].values, **p, color=cc)

        # Plot peak lines
        for t in dates[~np.isnan(df_track.peaks)]:
            ax.axvline(t, color='k', lw=1.2)

        # Set formatting
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        endDate = datetime(1970,1,1,3,45)

        ax.set_xlim(mpl.dates.date2num(startDate),mpl.dates.date2num(endDate))
        for h in hlines:
            ax.axhline(h,  color='k', lw=1)

        # Define legend titel
        dl = df_track.l2_ma.mean() / df_track.l1_ma.mean()
        s = r'$l_{short}/l_{long} = $' + '{:.2f}'.format(dl)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1,
                  frameon=False, title=s)


    # Format legend
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    axes = fig.get_axes()
    for ax in axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    # Define filename
    label = df_track.label.values[0]

    plt.show()
    filename = os.path.join(saveDir, "ts_several_lol.png".format(label))
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
