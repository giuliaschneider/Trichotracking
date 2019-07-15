import os.path
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ._hist import hist_aggregating, hist_breakup, hist_peak


def plot_thist(df, vcol, filename, xlabel, labels=None, text=None,
               xscale=None, yscale=None, report=False, fit=False,
               xfitstr=None, yfitstr=None, sigTest=False, cdf=False,
               legendMean=False, legendTimescale=False, meanL=None,
               plotMean=False):
    """ Saves a histogram of all v in vcol. """
    hist_aggregating(df, vcol, filename, xlabel, labels=labels,
                    text=text, xscale=xscale, yscale=yscale, report=report,
                    fit=fit, xfitstr=xfitstr, yfitstr=yfitstr,
                    minTh=1, sigTest=sigTest, cdf=cdf, meanL=meanL,
                    legendMean=legendMean, legendTimescale=legendTimescale,
                    plotMean=plotMean)


def plot_thist_breakup(df, vcol, filename, xlabel, labels=None, text=None,
                   labelAgg='Aggregating', xscale=None, yscale=None,
                   report=False, fit=False, xfitstr=None, yfitstr=None,
                   sigTest=False, legendMean=False, legendTimescale=False,
                   meanL=None, cdf=False, plotMean=False, kde=False):
    """ Saves a histogram of all v in vcol. """
    hist_breakup(df, vcol, filename, xlabel, labelAgg,labels=labels,
                 text=text, xscale=xscale, yscale=yscale, report=report,
                 fit=fit, xfitstr=xfitstr, yfitstr=yfitstr,
                 sigTest=sigTest, cdf=cdf, meanL=meanL,
                 legendMean=legendMean, legendTimescale=legendTimescale,
                 plotMean=plotMean, kde=kde)


def plot_thist_peak(df, vcol, filename, xlabel, labels=None, text=None,
                    xscale=None, yscale=None, report=False, fit=False,
                    xfitstr=None, yfitstr=None, sigTest=False, cdf=False,
                    legendMean=False, legendTimescale=False, meanL=None,
                    plotMean=False):
    """ Saves a histogram of all v in vcol. """
    hist_peak(df, vcol, filename, xlabel,labels=labels,
              text=text, xscale=xscale, yscale=yscale, report=report,
              fit=fit, xfitstr=xfitstr, yfitstr=yfitstr,
              sigTest=sigTest, cdf=cdf, meanL=meanL,
              legendMean=legendMean, legendTimescale=legendTimescale,
              plotMean=plotMean)


def plot_allthist(overlapDir, df, dflinked, df_tracks, mdfo, mdfs, meanL,
                  exp=None):
    if exp is None:
        exp = np.unique(mdfo.exp.values)

    if isinstance(exp, str):
        savedirexp = join(overlapDir, exp)
        if not os.path.isdir(savedirexp):
            os.mkdir(savedirexp)
        savedir = os.path.join(overlapDir, exp, 't')
        expfile = exp
        exp = [exp]
    else:
        savedir = join(overlapDir, 't')
        expfile = ''

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    savedirSingle = join(savedir, 'tsingle')
    if not os.path.isdir(savedirSingle):
        os.mkdir(savedirSingle)

    savedirb = join(savedir, 't_breakup_nobreakup')
    if not os.path.isdir(savedirb):
        os.mkdir(savedirb)

    savediragg = join(savedir, 't_aggregating_nonaggregating')
    if not os.path.isdir(savediragg):
        os.mkdir(savediragg)


    xlabel = r'$t$ [s]'
    xfitstr = 't'
    yfitstr = ''


    # Plot histogram of single filament time scales

    slabels = df_tracks[df_tracks.type==1].label
    filename = join(savedirSingle, expfile+"_hist_tsingle_cdf.png")
    plot_thist(mdfs[mdfs.exp.isin(exp)], 't', filename, xlabel,
               report=True, cdf=True, fit=True, sigTest=True,
               plotMean=True)
    """
    filename = join(savedirSingle, expfile+"_hist_tsingle_loglog.png")
    plot_thist(df_tracks[df_tracks.exp.isin(exp)], 't', filename, xlabel,
               slabels, xscale='log', yscale='log')
    filename = join(savedirSingle, expfile+"_hist_tsingle_semilog.png")
    plot_thist(df_tracks[df_tracks.exp.isin(exp)], 't', filename, xlabel,
               slabels, yscale='log')
    """


    # Plot histogram of filament pair time scales
    slabels = df_tracks[df_tracks.type==2].label
    filename = join(savedir, expfile +"_hist_tfilament_cdf.png")
    plot_thist(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel,
               report=True, cdf=True, fit=True, sigTest=True,
               plotMean=True)



    """
    filename = join(savedir, expfile +"_hist_tfilament_loglog.png")
    plot_thist(df_tracks[df_tracks.exp.isin(exp)], 't', filename, xlabel,
               slabels, xscale='log', yscale='log')
    filename = join(savedir, expfile +"_hist_tfilament_semilog.png")
    plot_thist(df_tracks[df_tracks.exp.isin(exp)], 't', filename, xlabel,
               slabels, yscale='log', report=True, fit=True,
               xfitstr=xfitstr, yfitstr=yfitstr, sigTest=True)
    """
    xlabel = r'$t$ [s]'
    xfitstr = 't'
    # Plot histogram of overlap velocities (breakup)
    """
    labels = mdfo[mdfo.breakup == 2].label.values
    filename = join(savedirb, expfile +"_hist_tfilament_breakup.png")
    plot_thist(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel, labels, text='Breakup')
    filename = join(savedirb, expfile +"_hist_tfilament_breakup_loglog.png")
    plot_thist(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel, labels, text='Breakup',
               xscale='log', yscale='log')
    filename = join(savedirb, expfile +"_hist_tfilament_breakup_semilog.png")
    plot_thist(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel, labels, text='Breakup',
               yscale='log')

    filename = join(savediragg, expfile +"_hist_tfilament_all.png")
    plot_thist_breakup(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel,
                       labelAgg='All', report=True)
    """
    filename = join(savediragg, expfile +"_hist_tfilament_split_cdf.png")
    plot_thist_breakup(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel,
                       labelAgg='', report=True, fit=True, sigTest=True,
                       xfitstr=xfitstr, yfitstr=yfitstr, plotMean=True,
                       cdf=True)
    slabels = mdfo[mdfo.aggregating].label

    filename = join(savediragg, expfile +"_hist_tfilament_split.png")
    plot_thist_breakup(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel,
                       labelAgg='Aggregating', report=True, legendMean=True,
                       xfitstr=xfitstr, yfitstr=yfitstr, plotMean=True,
                       kde=False, labels=slabels)
    """


    # Plot histogram of overlap velocities (agg)
    labels = mdfo[mdfo.breakup != 2].label.values
    filename = join(savedirb, expfile +"_hist_tfilament_nobreakup.png")
    plot_thist(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel,labels, text='No breakup')
    filename = join(savedirb, expfile +"_hist_tfilament_nobreakup_loglog.png")
    plot_thist(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel,labels, text='No breakup',
               xscale='log', yscale='log')
    filename = join(savedirb, expfile +"_hist_tfilament_nobreakup_semilog.png")
    plot_thist(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel,labels, text='No breakup',
               yscale='log')

    # Aggregating, breakup vs no breakup
    labels = mdfo[mdfo.aggregating == 1].label.values
    filename = join(savediragg, expfile +"_hist_tfilament_aggregating.png")
    plot_thist_breakup(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel,labels)
    filename = join(savediragg, expfile +"_hist_tfilament_aggregating_loglog.png")
    plot_thist_breakup(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel,labels,
               xscale='log', yscale='log')
    filename = join(savediragg, expfile +"_hist_tfilament_aggregating_semilog.png")
    plot_thist_breakup(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel, labels,
               yscale='log')


    # Non aggregating, LOL=0 vs LOL > 0
    labels = mdfo[mdfo.aggregating == 0].label.values
    filename = join(savediragg, expfile +"_hist_tfilament_nonaggregating.png")
    plot_thist_breakup(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel, labels,
                   labelAgg='Non aggregating')
    filename = join(savediragg, expfile +"_hist_tfilament_nonaggregating_loglog.png")
    plot_thist_breakup(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel, labels,
               labelAgg='Non aggregating', xscale='log', yscale='log')
    filename = join(savediragg, expfile +"_hist_tfilament_nonaggregating_semilog.png")
    plot_thist_breakup(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel, labels,
               labelAgg='Non aggregating', yscale='log')

    """

    # Peaks
    filename = join(savedir, expfile +"_hist_toverlap_reversals_cdf.png")
    plot_thist_peak(mdfo, 't', filename, xlabel,report=True,cdf=True,
                    fit=True, sigTest=True, plotMean=True)

    """
    filename = join(savedir, expfile +"_hist_toverlap_reversals_semilog.png")
    plot_thist_peak(mdfo, 't', filename, xlabel, yscale='log',report=True,
                    fit=True, xfitstr=xfitstr, yfitstr=yfitstr,
                    sigTest=True)
    """

    filename = join(savedir, expfile +"_hist_toverlap_reversals1_cdf.png")
    slabels = mdfo[mdfo.hasPeaks == 1].label.values
    plot_thist(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel, slabels,
               text='Reversing Pairs', report=True, cdf=True, fit=True,
               sigTest=True, plotMean=True)

    slabels = mdfo[mdfo.hasPeaks == 0].label.values
    filename = join(savedir, expfile +"_hist_toverlap_reversals2_cdf.png")
    plot_thist(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel, slabels,
               text='Non-reversing Pairs', report=True, fit=True,
               xfitstr=xfitstr, yfitstr=yfitstr, sigTest=True, cdf=True,
               plotMean=True)

    slabels = mdfo[mdfo.revb == 1].label.values
    filename = join(savedir, expfile +"_hist_toverlap_reversals3_cdf.png")
    plot_thist(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel, slabels,
               text='Reversing, non-separating pairs', report=True, fit=True,
               xfitstr=xfitstr, yfitstr=yfitstr, sigTest=True, cdf=True,
               plotMean=True)
    slabels = mdfo[mdfo.revb == 2].label.values
    filename = join(savedir, expfile +"_hist_toverlap_reversals4_cdf.png")
    plot_thist(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel, slabels,
               text='Reversing, separating pairs', report=True, fit=True,
               xfitstr=xfitstr, yfitstr=yfitstr, sigTest=True, cdf=True,
               plotMean=True)

    slabels = mdfo[mdfo.revb == 4].label.values
    filename = join(savedir, expfile +"_hist_toverlap_reversals5_cdf.png")
    plot_thist(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel, slabels,
               text='Non-reversing, separating pairs', report=True, fit=True,
               xfitstr=xfitstr, yfitstr=yfitstr, sigTest=True, cdf=True,
               plotMean=True)

    slabels = mdfo[mdfo.revb == 3].label.values
    filename = join(savedir, expfile +"_hist_toverlap_reversals6_cdf.png")
    plot_thist(mdfo[mdfo.exp.isin(exp)], 't', filename, xlabel, slabels,
               text='Non-reversing, non-separating pairs', report=True, fit=True,
               xfitstr=xfitstr, yfitstr=yfitstr, sigTest=True, cdf=True,
               plotMean=True)
