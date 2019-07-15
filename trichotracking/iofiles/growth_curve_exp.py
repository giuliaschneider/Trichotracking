import sys
import os
from os.path import abspath, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq
from logistic_growth import *

from IPython.core.debugger import set_trace


# Define variables
filepath = "/home/giu/Documents/10Masterarbeit/data/2018_10_growth_curve/"\
            "growth_curve_41A_2018_10_01.csv"
data = pd.read_csv(filepath, sep=";", skiprows=[1])
data['control'] = data[data.Day==25].cv2_cum_length

titles = ["Density", "Cumuluated length"]
ylabels = [r"fil/$\mu$L", r"mm/$\mu$L"]
allLabels = [["Manual count",], \
             ["Image analysis",]
             ]
allLegends = [True, True]
allMethods = [["count",], \
              ["fit_cum_length",]]
allFits = [[0.024], [0.007,]]
plotMethod = [plt.plot, plt.plot]


# Plot
f, axarr = plt.subplots(len(titles), figsize=(7,11), sharex=True)
f.subplots_adjust(hspace=0.3)

for ax, title, ylabel, methods, labels, fits, legend in zip(
    axarr, titles, ylabels, allMethods, allLabels, allFits, allLegends):
    # colors
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.arange(len(methods)-1,-1,-1)]
    plt.sca(ax)

    # Iterate through all method
    for pm, c, method, label, fit in zip(plotMethod, colors, methods, labels, fits):
        t = data.Day[~np.isnan(data[method])]
        p = data[method][~np.isnan(data[method])]
        weights = data["Dilution factor"][~np.isnan(data[method])]/5
        pm(t,np.log(p.values),'o', label=label, color=c)

        if fit is not None:
            t_fit = t[2:7].values
            p_fit = (p[2:7].values)
            kgrowth, x0 = np.polyfit((t_fit), np.log(p_fit),1)
            plt.plot(t_fit, kgrowth*t_fit +x0,'k--',linewidth=0.5)
            doubling_time = np.log(2)/kgrowth
            textstr = r"$r = %2.3f$"%(kgrowth) \
                        + "\nDoubling time = %2.1f d"%(doubling_time)
            pos = (0.4, 0.3)
            ax.annotate(textstr, xy=pos, xytext=pos, textcoords='axes fraction',color=c)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()

#axarr[2].set_ylim([0,1])
axarr[-1].set_xlabel("Time (d)")
fname = "/home/giu/Documents/10Masterarbeit/results/growth_curve/" \
        "2018_10_01_41A_growth_curve_exp"
saveAsTex(f, fname)
