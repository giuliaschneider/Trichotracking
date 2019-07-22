import os.path
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ._hist import hist_aggregating, hist_breakup, hist_lol,hist_peak, getHistData


def plot_peakhist(df, vcol, filename, xlabel, labels=None, text=None,
               xscale=None, yscale=None, cdf=False, maxTh=None, minTh=None,
               xlim=None, plotMean=False, bins=None, report=False,
               legend_out=True, appAgglabel='', appNaggLabel='',
               sigTest=True, fit=True, xfitstr=None, yfitstr=None):
    """ Saves a histogram of all v in vcol. """
    hist_aggregating(df, vcol, filename, xlabel, labels=labels, maxTh=maxTh,
                    text=text, xscale=xscale, yscale=yscale,
                    minTh=minTh, xlim=xlim, plotMean=plotMean, bins=bins,
                    report=report, legend_out=legend_out,
                    appAgglabel=appAgglabel, appNaggLabel=appNaggLabel,
                    cdf=cdf, sigTest=sigTest, fit=fit, xfitstr=xfitstr,
                    yfitstr=yfitstr)




def plot_allpeakhist(overlapDir, df, dflinked, df_tracks, mdfo, mdfs, meanL,
                     exp=None):
    if exp is None:
        exp = np.unique(mdfo.exp.values)

    if isinstance(exp, str):
        savedirexp = join(overlapDir, exp)
        if not os.path.isdir(savedirexp):
            os.mkdir(savedirexp)
        savedir = os.path.join(overlapDir, exp, 'peaks')
        expfile = exp
        exp = [exp]
    else:
        savedir = join(overlapDir, 'peaks')
        expfile = ''

    if not os.path.isdir(savedir):
        os.mkdir(savedir)


    xlabel = r'$n_{peaks}$'
    filename = join(savedir, expfile +"_hist_peaks.png")
    bins = np.arange(-0.5, mdfo.peaks.max()+0.5).tolist()
    plot_peakhist(mdfo[mdfo.exp.isin(exp)], 'peaks', filename, xlabel,
                  plotMean=True, bins=bins, report=True, cdf=True,
                  fit=True, yfitstr='', xfitstr='n_p', sigTest=True)
    plot_peakhist(mdfo[mdfo.exp.isin(exp)], 'peaks', filename, xlabel,
                  plotMean=True, bins=bins, report=True, cdf=True,
                  fit=True, yfitstr='', xfitstr='n_p', sigTest=True)
    """
    filename = join(savedir, expfile +"_hist_peaks_semilog.png")
    plot_peakhist(mdfo[mdfo.exp.isin(exp)], 'peaks', filename, xlabel,
                  plotMean=True, bins=bins, report=True, yscale='log')
    """
    """
    xlabel = r'$T_{reversal}$ [s]'
    revFa = mdfs[mdfs.aggregating].hasPeaks.sum()/mdfs[mdfs.aggregating].hasPeaks.count()
    revFna = mdfs[~mdfs.aggregating].hasPeaks.sum()/mdfs[~mdfs.aggregating].hasPeaks.count()
    appAgglabel = ', {:.0f}\% reversing'.format(revFa*100)
    appNaggLabel= ', {:.0f} \% reversing'.format(revFna*100)
    filename = join(savedir, expfile +"_hist_trev_single.png")
    slabels = df_tracks[df_tracks.type==1].label.values
    plot_peakhist(mdfs[mdfs.exp.isin(exp)], 'trev', filename,
                  xlabel, labels=slabels, report=True,
                   appAgglabel=appAgglabel, appNaggLabel=appNaggLabel)
    filename = join(savedir, expfile +"_hist_trev_single_semilog.png")
    plot_peakhist(mdfs[mdfs.exp.isin(exp)], 'trev', filename,
                  xlabel, report=True, yscale='log', labels=slabels,
                  appAgglabel=appAgglabel, appNaggLabel=appNaggLabel)
    """
    """
    revFa = mdfo[mdfo.aggregating].hasPeaks.sum()/mdfo[mdfo.aggregating].hasPeaks.count()
    revFna = mdfo[~mdfo.aggregating].hasPeaks.sum()/mdfo[~mdfo.aggregating].hasPeaks.count()
    appAgglabel = ', {:.0f}\% reversing'.format(revFa*100)
    appNaggLabel= ', {:.0f} \% reversing'.format(revFna*100)
    filename = join(savedir, expfile +"_hist_trev_pairs.png")
    plot_peakhist(mdfo[mdfo.exp.isin(exp)], 'trev', filename, xlabel,
                  plotMean=True, report=True)
    filename = join(savedir, expfile +"_hist_trev_pairs_semilog.png")
    plot_peakhist(mdfo[mdfo.exp.isin(exp)], 'trev', filename, xlabel,
                  report=True, yscale='log',
                  appAgglabel=appAgglabel, appNaggLabel=appNaggLabel)
    """
