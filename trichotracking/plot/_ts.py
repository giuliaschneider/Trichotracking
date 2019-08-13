import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.debugger import set_trace

# Time series settings
from matplotlib import cm

props_marker = {'ls': '', 'marker': 'o', 'ms': 4}
props_line = {'ls': '-', 'marker': ''}
xlabel = ['Time [hh:mm]']


__all__ = ['ts', 'ts_oneQuantity']

def ts_oneQuantity(df, col, ylabel, filename):
    ts(df, [(col,)], [ylabel], [(cm.tab10(0),)], [(props_line,)], [(ylabel,)], filename)


def ts(df, cols, labels, colors, props, ylabels, filename,
       ylims=None, hlines=None, title=None,
       darkphases=None, texts=None, legTitles=None,
       figsize=None):
    """ Plots length/overlap time series for t(nAxis,1,1) """

    # Set up figure
    mpl.rcParams.update({'font.size': 12})
    nAxis = len(cols)
    if figsize is None:
        figsize = (14, 6)
    fig, axes = plt.subplots(nAxis, 1, sharex=True, figsize=figsize)
    if nAxis == 1:
        axes = [axes]
    if title is not None:
        axes[0].set_title(title)
    dates = mpl.dates.epoch2num(df.time.values)

    # Get darkphases
    if darkphases is not None:
        darkphases = None
    #    dark = getDarkPhases(darkphases, dates)

    if texts is None:
        texts = [None for i in range(nAxis + 1)]

    if legTitles is None:
        legTitles = [None for i in range(nAxis + 1)]

    if hlines is None:
        hlines = [None for i in range(nAxis + 1)]

    if ylims is None:
        ylims = [None for i in range(nAxis + 1)]

    # Iterate through all subplots
    for i in range(nAxis):
        ax = axes[i]
        if i < nAxis - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        col = cols[i]
        color = colors[i]
        prop = props[i]
        label = labels[i]

        # Plot data
        for co, pr, la, cco in zip(col, prop, label, color):
            ax.plot_date(dates, df[co], label=la, **pr, color=cco)

        # Plot peak lines
        if 'peaks' in df.keys():
            for t in dates[~np.isnan(df.peaks)]:
                ax.axvline(t, color='k', lw=1.2)


        # Set formatting
        ax.set_ylabel(ylabels[i])
        if ylims[i] is not None:
            ax.set_ylim(ylims[i])
        if hlines[i] is not None:
            for h in hlines[i]:
                ax.axhline(h, color='k', lw=1)
        if texts[i] is not None:
            ax.text(**texts[i], transform=ax.transAxes)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1,
                  frameon=False, title=legTitles[i])

    # Format legend
    ax = axes[-1]
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
