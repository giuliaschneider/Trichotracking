import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms
from matplotlib.collections import LineCollection
from matplotlib import cm
from overlap.analysis_overlap import *

mpl.rcParams.update({'font.size': 12})


def plotOverlapPhase(df, dfMeta, col, vcol, saveDir, xlabel, ylabel,
    xlim=None, ylim=None, filename_appe=None):
    """ Plots df.col vs df.vcol in phasespace. """

    expLabels = np.unique(dfMeta.exp)

    # Iterate all experimetns
    for exp in expLabels:
        trackLabels = np.unique(df[df.exp==exp].label)
        dt = dfMeta[dfMeta.exp==exp].dt.values[0]

        # Iterate all labels
        for l in trackLabels:
            df_track = df[df.label == l]
            # Define title
            track = int(df_track.track.values[0])
            title = "Track = {}".format(track)

            # Define filename
            label = df_track.label.values[0]
            saveDirexp = os.path.join(saveDir, exp)
            filename = os.path.join(saveDirexp, "{}_phase".format(label))
            if filename_appe is not None:
                filename += filename_appe

            # Define data
            x = df_track[col].values
            y = df_track[vcol].values
            isnan = np.isnan(x) | np.isnan(y)
            x = x[~isnan]
            y = y[~isnan]
            t = np.arange(x.size)*dt

            if x.size > 0:
                # Define colored segments
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)


                # Plot
                fig, ax = plt.subplots(1, 1)
                norm = plt.Normalize(t.min(), t.max())
                lc = LineCollection(segments, cmap='copper', norm=norm)
                lc.set_array(t)
                line = ax.add_collection(lc)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                ax.set_title(title)

                if xlim is not None:
                    ax.set_xlim(xlim)
                else:
                    ax.set_xlim(0.9*x.min(), 1.1*x.max())

                if ylim is not None:
                    ax.set_ylim(ylim)
                else:
                    ax.set_ylim(0.9*y.min(), 1.1*y.max())
                cbar = plt.colorbar(line)
                cbar.set_label('Time [s]')
                fig.savefig(filename, bbox_inches='tight', dpi=300)
                plt.close(fig)
