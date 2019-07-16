import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms
from matplotlib.collections import LineCollection
from matplotlib import cm
from overlap.analysis_overlap import *

mpl.rcParams.update({'font.size': 10})


def plotFrequencies(df, dfMeta, saveDir):
    """ Plots frequency spectrum of track time series. """

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
            filename = os.path.join(saveDirexp, "{}_f".format(label))

            # Calc frequencies
            f, freq = calcFrequencies(df_track, dt)

            # Calc max frequency
            fabs = np.abs(f)
            fmax = fabs.max()
            freqMajor = freq[(fabs>0.15*fmax) & (freq>0)]
            fMajor = fabs[(fabs>0.15*fmax) & (freq>0)]

            """if label == '0050360':
                set_trace()"""

            if fMajor.size > 0 :

                # Plot
                fig = plt.figure()
                tfreq = np.divide(1, freq, out=np.zeros_like(freq), where=freq!=0)
                plt.plot(freq, fabs)
                plt.xlim([-0.01, 0.01])
                plt.xlabel("Reversal frequency [Hz]")
                plt.ylabel("DFT values")
                plt.title("Track = {}".format(label))

                ind = np.argmax(fMajor)
                freqMajor  = freqMajor[ind]
                fMajor = fMajor[ind]
                trev = 1/freqMajor/60
                strText = r"$T_{reverse} = $"+"{:.1f} min".format(trev)
                plt.text(freqMajor, fMajor, strText, horizontalalignment='left')

                fig.savefig(filename, bbox_inches='tight', dpi=150)
                plt.close(fig)
