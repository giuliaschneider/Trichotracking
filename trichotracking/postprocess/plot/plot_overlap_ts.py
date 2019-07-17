import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms
from matplotlib import cm
from IPython.core.debugger import set_trace


# Time series settings
props_marker = {'ls':'', 'marker': 'o', 'ms':4}
props_line = {'ls':'-', 'marker':''}
xlabel = ['Time [hh:mm]']


def plotTracksV(df, dfMeta, saveDir):
    columns = [("v1_ma", "v2_ma","v_pos_ma"),
               ("a1", "a2")
               ]
    labels = [(r"$v_1$", r"$v_2$", r"$v_{pos}$"),
              (r"$a_1$", r"$a_2$",)]
    props = [(props_line, props_line, props_line,),
             (props_line, props_line)]
    ylabels = [r'$v$ [µm/s]',
               r'$a$ [µm/s$^2$]']
    ylims = [(-3, 3),
             (-0.03, 0.03)]
    #hlines = [(0,),]
    colors = [(cm.tab10(2), cm.tab10(8), cm.tab10(5)),
              (cm.tab10(2), cm.tab10(8)),]
    exp = df.exp[~df.exp.isnull()].values[0]
    label = df.label[~df.l1_ma.isnull()].values[0]
    agginducer = dfMeta[dfMeta.exp==exp].agginducer.values[0]

    if agginducer == 'light':
        darkphases = dfMeta[dfMeta.exp==exp].darkphases.values[0]
    else:
        darkphases = None

    # Define legend titel
    dl = df.l2_ma.mean() / df.l1_ma.mean()
    s = r'$l_{short}/l_{long} = $' + '{:.2f}'.format(dl)
    legT = [s, None ]

    # Define filename
    saveDirexp = os.path.join(saveDir, exp)
    filename = os.path.join(saveDirexp, "{}_ts_v.png".format(label))

    # Plot
    plotOverlapTimeSeries(df, columns, labels, colors, props,
        ylabels, xlabel, ylims, None, None, filename,
        darkphases=darkphases, legTitles=legT, figsize=(14,6/2))


def plotTracksLOL(df, dfMeta, saveDir):
    """ Plots the LOL time series for all tracks in df."""


    columns = [("xlov_norm", "xlov_ma",),]
    labels = [(r"$LOL$", r"$LOL_{ma}$",),]
    props = [(props_marker, props_line,),]
    ylabels = [r'$LOL$',]
    ylims = [(-1, 1),]
    hlines = [(0,),]
    colors = [(cm.tab20(8), cm.tab20(9)),]

    expLabels = np.unique(dfMeta.exp)

    # Iterate all experimetns
    for exp in expLabels:
        trackLabels = np.unique(df[df.exp==exp].label)
        dt = dfMeta[dfMeta.exp==exp].dt
        darkphases = dfMeta[dfMeta.exp==exp].darkphases.values[0]

        # Iterate all labels
        for l in trackLabels:
            df_track = df[df.label == l]

            # Define title
            track = int(df_track.track.values[0])
            title = "Track = {}".format(track)

            # Define legend titel
            dl = df_track.l2_ma.mean() / df_track.l1_ma.mean()
            s = r'$l_{short}/l_{long} = $' + '{:.2f}'.format(dl)
            legT = [s, ]

            # Define filename
            label = df_track.label.values[0]
            saveDirexp = os.path.join(saveDir, exp)
            filename = os.path.join(saveDirexp, "{}_ts_lol.png".format(label))

            # Plot
            plotOverlapTimeSeries(df_track, columns, labels, colors, props,
                ylabels, xlabel, ylims, hlines, title, filename,
                darkphases=darkphases, legTitles=legT, figsize=(14,6/4))


def plotTracks(df,  dfMeta, saveDir):
    """ Plots thetime series for all tracks in df."""

    # Time series settings
    props_marker = {'ls':'', 'marker': 'o', 'ms':4}
    props_line = {'ls':'-', 'marker':''}
    xlabel = ['Time [hh:mm]']

    columns = [("length1", "l1_ma", "length2", "l2_ma"),
               ("xlov_norm", "xlov_ma",),
               ("pos_ma",),
               #("v1", "v2", "v_pos", "v2_v1"),
               ("v_pos",)
               ]
    labels = [(r"$l_{1}$", r"$l_{1,ma}$", r"$l_{2}$", r"$l_{2,ma}$"),
              (r"$LOL$", r"$LOL_{ma}$",),
              ("Position",),
              #(r"$v_1$", r"$v_2$", r"$\Delta$ Pos", r"$v_2-v_1$"),
              ("v_pos",)
              ]
    props = [(props_marker, props_line, props_marker, props_line),
             (props_marker, props_line,),
             (props_marker,),
             #(props_line, props_line, props_line, props_line),
             (props_line,)
             ]
    ylabels = [r'$l$ [µm]',
               r'$LOL$',
               r'$Pos$',
               #r'$v$ [µn/s]',
               r'$v$ [µm/s]',
               ]
    ylims = [None,
             (-1, 1),
             None, #(-6, 7),
             #(-3, 3),
             (-4, 4)]

    hlines = [None,
             (0,),
             (0, 1,),
             #None,
             None,]

    colors = [(cm.tab20(0), cm.tab20(1), cm.tab20(2), cm.tab20(3)),
              (cm.tab20(8), cm.tab20(9)),
              (cm.tab20(14),),
              #(cm.tab10(2), cm.tab10(8), cm.tab10(5),  cm.tab10(6))
              (cm.tab10(2),)]

    expLabels = np.unique(dfMeta.exp)

    # Iterate all experimetns
    for exp in expLabels:
        trackLabels = np.unique(df[df.exp==exp].label)
        dt = dfMeta[dfMeta.exp==exp].dt
        darkphases = dfMeta[dfMeta.exp==exp].aggchambers.values[0]

        # Iterate all labels
        for l in trackLabels:
            df_track = df[df.label == l]

            # Define title
            track = int(df_track.track.values[0])
            dl = df_track.l2_ma.mean() / df_track.l1_ma.mean()
            title = "Track = {}".format(track)

            # Define filename
            label = df_track.label.values[0]
            saveDirexp = os.path.join(saveDir, exp)
            filename = os.path.join(saveDir, "{}_ts.png".format(label))

            # Plot
            plotOverlapTimeSeries(df_track, columns, labels, colors, props,
                ylabels, xlabel, ylims, hlines, title, filename,
                darkphases=darkphases, figsize=(14,6))



def plotOverlapTimeSeries(df_track, cols, labels, colors, props, ylabels,
                          xlabel, ylims, hlines, title, filename,
                          darkphases=None, texts=None, legTitles=None,
                          figsize=None):
    """ Plots length/overlap time series for t(nAxis,1,1) """

    # Set up figure
    mpl.rcParams.update({'font.size': 12})
    nAxis = len(cols)
    if figsize is None:
        figsize = (14,6)
    fig, axes = plt.subplots(nAxis, 1, sharex=True, figsize=figsize)
    if nAxis == 1:
        axes = [axes]
    if title is not None:
        axes[0].set_title(title)
    dates = mpl.dates.date2num(df_track.time.values)

    # Get darkphases
    if darkphases is not None:
        darkphases=None
    #    dark = getDarkPhases(darkphases, dates)

    if texts is None:
        texts = [None for i in range(nAxis+1)]

    if legTitles is None:
        legTitles = [None for i in range(nAxis+1)]

    if hlines is None:
        hlines = [None for i in range(nAxis+1)]

    if ylims is None:
        ylims = [None for i in range(nAxis+1)]

    # Iterate through all subplots
    for i in range(nAxis):
        ax = axes[i]
        if i < nAxis-1:
            plt.setp(ax.get_xticklabels(), visible=False)
        col = cols[i]
        color = colors[i]
        prop = props[i]
        label = labels[i]

        # Plot data
        for c, p, l, cc in zip(col, prop, label, color):
            ax.plot_date(dates, df_track[c], label=l, **p, color=cc)

        # Plot peak lines
        for t in dates[~np.isnan(df_track.peaks)]:
            ax.axvline(t, color='k', lw=1.2)

        # Shade dark areas
        if darkphases is not None:
            trans = mtransforms.blended_transform_factory(ax.transData,
                                                          ax.transAxes)
            ax.fill_between(dates, 0, 1, where=(dark),transform=trans,
                             facecolor=cm.tab20(15), alpha=0.7)

        # Set formatting
        ax.set_ylabel(ylabels[i])
        if ylims[i] is not None:
            ax.set_ylim(ylims[i])
        if hlines[i] is not None:
            for h in hlines[i]:
                ax.axhline(h,  color='k', lw=1)
        if texts[i] is not None:
            ax.text(**texts[i],  transform=ax.transAxes)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1,
                  frameon=False, title=legTitles[i])


    # Format legend
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    axes = fig.get_axes()
    for ax in axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    try:
        fig.savefig(filename, bbox_inches='tight', dpi=300)
    except:
        set_trace()
    plt.close(fig)
