import os.path
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ._hist import hist_aggregating, hist_breakup, hist_all


def plot_lolhist(df, vcol, filename, xlabel, labels=None, text=None,
               xscale=None, yscale=None, report=False,fit=False,
               xfitstr=None, yfitstr=None, sigTest=False, cdf=False,
               legendMean=False, legendTimescale=False, meanL=None,
               plotMean=False, all=False, minTh=None, xlim=None,
               left=False):
    """ Saves a histogram of all v in vcol. """
    if all:
        hist_all(df, vcol, filename, xlabel, labels=labels, minTh=minTh,
                        text=text, xscale=xscale, yscale=yscale, report=report,
                        sigTest=sigTest, cdf=cdf, meanL=meanL, fit=fit,
                        legendMean=legendMean, legendTimescale=legendTimescale,
                        plotMean=plotMean, xfitstr=xfitstr, yfitstr=yfitstr,
                        xlim=xlim)
    else:
        hist_aggregating(df, vcol, filename, xlabel, labels=labels, minTh=None,
                        text=text, xscale=xscale, yscale=yscale, report=report,
                        sigTest=sigTest, cdf=cdf, meanL=meanL, fit=fit,
                        legendMean=legendMean, legendTimescale=legendTimescale,
                        plotMean=plotMean, xfitstr=xfitstr, yfitstr=yfitstr,
                        xlim=xlim, left=left)


def plot_lolhist_breakup(df, vcol, filename, xlabel, labels=None, text=None,
                   labelAgg='Aggregating', xscale=None, yscale=None,
                   report=False,fit=False,
                   xfitstr=None, yfitstr=None, sigTest=False, cdf=False,
                   legendMean=False, legendTimescale=False, meanL=None,
                   plotMean=False, kde=False):
    """ Saves a histogram of all v in vcol. """
    hist_breakup(df, vcol, filename, xlabel, labelAgg,labels=labels,
                 minTh=None, text=text, xscale=xscale, yscale=yscale,
                 report=report, sigTest=sigTest, cdf=cdf, meanL=meanL,
                 legendMean=legendMean, legendTimescale=legendTimescale,
                 plotMean=plotMean, xfitstr=xfitstr, yfitstr=yfitstr,
                 fit=fit, kde=kde)




def plot_alllolhist(overlapDir, df, dflinked, df_tracks, mdfo, meanL,
                    exp=None):
    if exp is None:
        exp = np.unique(mdfo.exp.values)

    if isinstance(exp, str):
        savedirexp = join(overlapDir, exp)
        if not os.path.isdir(savedirexp):
            os.mkdir(savedirexp)
        savedir = join(overlapDir, exp, 'lol')
        expfile = exp
        exp = [exp]
    else:
        savedir = join(overlapDir, 'lol', 'hist')
        expfile = ''

    if not os.path.isdir(savedir):
        os.mkdir(savedir)


    savediragg = join(savedir, 'lol_aggregating_nonaggregating')
    if not os.path.isdir(savediragg):
        os.mkdir(savediragg)

    savedirb = join(savedir, 'lol_breakup_nobreakup')
    if not os.path.isdir(savedirb):
        os.mkdir(savedirb)

    xlabel = r'$|LOL_{rev}|$'
    xfitstr = '|LOL|'
    yfitstr = ''

    # Plot histogram of lol
    """
    filename = join(savedir, expfile +"_hist_lol.png")
    plot_lolhist(df[df.exp.isin(exp)], 'xlov_ma_abs', filename, xlabel)
    filename = join(savedir, expfile +"_hist_lol_loglog.png")
    plot_lolhist(df[df.exp.isin(exp)], 'xlov_ma_abs', filename, xlabel,
                 xscale='log', yscale='log')
    filename = join(savedir, expfile +"_hist_lol_semilog.png")
    plot_lolhist(df[df.exp.isin(exp)], 'xlov_ma_abs', filename, xlabel,
                 xscale='log', yscale='log')
    """
    slabels = mdfo[(mdfo.hasPeaks)].label.values

    df['ylov_n'] = (df.ylov / df.length2).abs()
    df['lol_rev'] = df[['xlov_ma_abs', 'ylov_n']].max(axis=1)
    df.loc[df.peaks.isnull(), 'lol_rev'] = np.nan
    df['lol_rev_abs'] = df.lol_rev * df.l2_ma
    filename = join(savedirb, expfile +"_hist_lol_all_cdf.png")
    plot_lolhist(df[df.exp.isin(exp)], 'lol_rev', filename,
                 xlabel, report=True, cdf=True, plotMean=True,
                 sigTest=True, xfitstr=xfitstr, legendMean=True,
                 yfitstr=yfitstr, all=True)


    # Histogram of lol peaks
    filename = join(savedir, expfile +"_hist_lol_cdf.png")
    plot_lolhist(df[df.exp.isin(exp)], 'xlov_ma_abs_peaks', filename,
                 xlabel, report=True, cdf=True, plotMean=True,
                 sigTest=True, xfitstr=xfitstr, legendMean=True,
                 yfitstr=yfitstr)

    slabels = mdfo[(mdfo.revb == 1)].label.values
    filename = join(savedir, expfile +"_hist_lol_cdf1.png")
    plot_lolhist(df[df.exp.isin(exp)], 'lol_rev', filename,
                 xlabel, report=True, cdf=True, plotMean=True,
                 sigTest=True, xfitstr=xfitstr, legendMean=True,
                 yfitstr=yfitstr, labels=slabels,
                 text='Reversing, non-separating pairs')
    slabels = mdfo[(mdfo.revb == 2)].label.values
    filename = join(savedir, expfile +"_hist_lol_cdf2.png")
    plot_lolhist(df[df.exp.isin(exp)], 'lol_rev', filename,
                 xlabel, report=True, cdf=True, plotMean=True,
                 sigTest=True, xfitstr=xfitstr, legendMean=True,
                 yfitstr=yfitstr, labels=slabels,
                 text='Reversing, separating pairs')


    # Absolute LOL
    slabels = mdfo[(mdfo.revb == 1)].label.values
    xlabel = r'$|x_{LOL}|$ [µm]'
    xfitstr = '|x_{LOL}|'
    filename = join(savedirb, expfile +"_hist_lol_abs_cdf1.png")
    plot_lolhist(df[df.exp.isin(exp)], 'lol_rev_abs', filename,
                 xlabel, report=True, cdf=True, plotMean=True,
                 sigTest=True, xfitstr=xfitstr, legendMean=True,
                 yfitstr=yfitstr, labels=slabels,
                 text='Reversing, non-separating pairs')
    slabels = mdfo[(mdfo.revb == 2)].label.values
    filename = join(savedirb, expfile +"_hist_lol_abs_cdf2.png")
    plot_lolhist(df[df.exp.isin(exp)], 'lol_rev_abs', filename,
                 xlabel, report=True, cdf=True, plotMean=True,
                 sigTest=True, xfitstr=xfitstr, legendMean=True,
                 yfitstr=yfitstr, labels=slabels,
                 text='Reversing, separating pairs')

    # Overlap
    xlabel = r'$l_{ov}$ [µm]'
    xfitstr = '|l_{ov}|'

    filename = join(savedir, expfile +"_hist_overlap.png")
    plot_lolhist(df[df.exp.isin(exp)&(df.peaks==1)], 'lov_ma', filename,
                 xlabel, report=True, cdf=True, plotMean=True,
                 sigTest=True, xfitstr=xfitstr, legendMean=True,
                 yfitstr=yfitstr)
    xlabel = r'$l_{ov}/l_2$'
    xfitstr = r'\frac{l_{ov}}{l_2}'
    df['lov_n'] = df.lov_ma / df.l2_ma
    df.loc[df.lov_n>1, 'lov_n'] = 1
    filename = join(savedir, expfile +"_hist_overlap_n.png")
    plot_lolhist(df[df.exp.isin(exp)&(df.peaks==1)], 'lov_n', filename,
                 xlabel, report=True, cdf=True, plotMean=True,
                 sigTest=True, xfitstr='x', legendMean=True,
                 yfitstr=yfitstr, xlim=(0,1), left=True)

    """
    slabels = mdfo[(mdfo.hasPeaks)].label.values
    filename = join(savedirb, expfile +"_hist_lol_reversing_cdf.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'lol_rev', filename,
                 xlabel, report=True, cdf=True, plotMean=True,
                 sigTest=True, xfitstr=xfitstr, legendMean=True,
                 yfitstr=yfitstr, labels=slabels,labelAgg='',
                 text='Reversing pairs')

    slabels = mdfo[(mdfo.hasPeaks)].label.values
    filename = join(savedirb, expfile +"_hist_lol_reversing.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'lol_rev', filename,
                 xlabel, report=True, kde=True, plotMean=True,
                 sigTest=True, xfitstr=xfitstr, legendMean=True,
                 yfitstr=yfitstr, labels=slabels,
                 text='Reversing pairs')
    """

    """
    filename = join(savedir, expfile +"_hist_lol_peaks_loglog.png")
    plot_lolhist(df[df.exp.isin(exp)], 'xlov_ma_abs_peaks', filename, xlabel,
                 xscale='log', yscale='log')
    filename = join(savedir, expfile +"_hist_lol_peaks_semilog.png")
    plot_lolhist(df[df.exp.isin(exp)], 'xlov_ma_abs_peaks', filename, xlabel,
                 yscale='log')
    """

    """


    # All

    filename = join(savedirb, expfile +"_hist_lol_all_peaks.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'xlov_ma_abs_peaks',
                        filename, xlabel, labelAgg='All')

    # Aggregating, breakup vs no breakup
    labels = mdfo[mdfo.aggregating == 1].label.values
    filename = join(savediragg, expfile +"_hist_lol_aggregating.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'xlov_ma_abs', filename, xlabel,labels)
    filename = join(savediragg, expfile +"_hist_lol_aggregating_loglog.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'xlov_ma_abs', filename,
                         xlabel, xscale='log', yscale='log')
    filename = join(savediragg, expfile +"_hist_lol_aggregating_semilog.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'xlov_ma_abs', filename,
                         xlabel, yscale='log')
    # Aggregating, breakup vs no breakup, peaks
    labels = mdfo[mdfo.aggregating == 1].label.values
    filename = join(savediragg, expfile +"_hist_lol_aggregating_peaks.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'xlov_ma_abs_peaks',
                         filename, xlabel,labels)
    filename = join(savediragg, expfile +"_hist_lol_aggregating_peaks_loglog.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'xlov_ma_abs_peaks',
                         filename, xlabel, labels, xscale='log', yscale='log')
    filename = join(savediragg, expfile +"_hist_lol_aggregating_peaks_semilog.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'xlov_ma_abs_peaks',
                         filename, xlabel, labels, yscale='log')


    # non aggregating, breakup vs no breakup
    labels = mdfo[mdfo.aggregating == 0].label.values
    filename = join(savediragg, expfile +"_hist_lol_nonaggregating.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'xlov_ma_abs', filename, xlabel,labels,
                   labelAgg='Non aggregating')
    filename = join(savediragg, expfile +"_hist_lol_nonaggregating_loglog.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'xlov_ma_abs', filename, xlabel,labels,
                   labelAgg='Non aggregating', xscale='log', yscale='log')
    filename = join(savediragg, expfile +"_hist_lol_nonaggregating_semilog.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'xlov_ma_abs', filename, xlabel,labels,
                   labelAgg='Non aggregating', yscale='log')

    # non aggregating, breakup vs no breakup, peaks
    labels = mdfo[mdfo.aggregating == 0].label.values
    filename = join(savediragg, expfile +"_hist_lol_nonaggregating_peaks.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'xlov_ma_abs_peaks',
                        filename, xlabel,labels, labelAgg='Non aggregating')
    labels = mdfo[mdfo.aggregating == 0].label.values
    filename = join(savediragg, expfile +"_hist_lol_nonaggregating_peaks_loglog.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'xlov_ma_abs_peaks',
                        filename, xlabel,labels, labelAgg='Non aggregating',
                        xscale='log', yscale='log')
    labels = mdfo[mdfo.aggregating == 0].label.values
    filename = join(savediragg, expfile +"_hist_lol_nonaggregating_peaks_semilog.png")
    plot_lolhist_breakup(df[df.exp.isin(exp)], 'xlov_ma_abs_peaks',
                        filename, xlabel,labels, labelAgg='Non aggregating',
                        yscale='log')
    """

    # Plot histogram of t lol
    xlabel = r'$T_{LOL}$ [s]'
    filename = join(savedir, expfile +"_hist_tlol_cdf.png")
    plot_lolhist(df[df.exp.isin(exp)], 'tlol_peaks', filename, xlabel,
                 report=True, cdf=True, plotMean=True,
                 sigTest=True, xfitstr='T_{LOL}', fit=True,
                 yfitstr=yfitstr)


    """
    filename = join(savedir, expfile +"_hist_tlol.png")
    plot_lolhist(df[df.exp.isin(exp)], 'tlol_peaks', filename, xlabel,
                 report=True)
    filename = join(savedir, expfile +"_hist_tlol_loglog.png")
    plot_lolhist(df[df.exp.isin(exp)], 'tlol_peaks', filename, xlabel,
                 xscale='log', yscale='log')
    filename = join(savedir, expfile +"_hist_tlol_semilog.png")
    plot_lolhist(df[df.exp.isin(exp)], 'tlol_peaks', filename, xlabel,
                 yscale='log', report=True)
    """


    # Plot histogram of t lol normed 1
    xlabel = r'$x_{LOL} / (l_1 + l_2 -l_{ov})$'
    xfitstr = r'\frac{x_{LOL}}{l_1 + l_2 -l_{ov}}'
    df['lol_normed1'] = df.xlov_abs_ma.abs() / (df.l1_ma + df.l2_ma - df.lov_ma)
    df['lol_normed2'] = df.xlov_abs_ma.abs() / (df.l1_ma + df.l2_ma)
    df.loc[df.peaks==0, 'lol_normed1'] = np.nan
    df.loc[df.peaks==0, 'lol_normed2'] = np.nan
    filename = join(savedir, expfile +"_hist_lol1_cdf.png")
    plot_lolhist(df[df.exp.isin(exp)], 'lol_normed1', filename, xlabel,
                report=True, cdf=True, plotMean=True,
                sigTest=True, xfitstr='x', legendMean=True,
                yfitstr=yfitstr)
    """
    filename = join(savedir, expfile +"_hist_lol1_loglog.png")
    plot_lolhist(df[df.exp.isin(exp)], 'lol_normed1', filename, xlabel,
                 xscale='log', yscale='log', minTh=0)
    filename = join(savedir, expfile +"_hist_lol1_semilog.png")
    plot_lolhist(df[df.exp.isin(exp)], 'lol_normed1', filename, xlabel,
                 yscale='log', minTh=0)
    """
    # Plot histogram of t lol normed 2
    xlabel = r'$x_{LOL}/ (l_1 + l_2)$'
    xfitstr = r'\frac{x_{LOL}}{l_1 + l_2}'
    filename = join(savedir, expfile +"_hist_lol2_cdf.png")
    plot_lolhist(df[df.exp.isin(exp)], 'lol_normed2', filename, xlabel,
                 report=True, cdf=True, plotMean=True,
                 sigTest=True, xfitstr='x', legendMean=True,
                 yfitstr=yfitstr)
    """
    filename = join(savedir, expfile +"_hist_lol2_loglog.png")
    plot_lolhist(df[df.exp.isin(exp)], 'lol_normed2', filename, xlabel,
                 xscale='log', yscale='log', minTh=0)
    filename = join(savedir, expfile +"_hist_lol2_semilog.png")
    plot_lolhist(df[df.exp.isin(exp)], 'lol_normed2', filename, xlabel,
                 yscale='log', minTh=0)

    # Plot histogram of overlap velocities (breakup)
    xlabel = r'$|LOL|$'
    labels = mdfo[mdfo.breakup == 2].label.values
    filename = join(savedirb, expfile +"_hist_lol_breakup.png")
    plot_lolhist(df[df.exp.isin(exp)], 'xlov_ma_abs', filename, xlabel,
                 labels, text='Breakup')
    # Plot histogram of overlap velocities (agg)
    labels = mdfo[mdfo.breakup != 2].label.values
    filename = join(savedirb, expfile +"_hist_lol_nobreakup.png")
    plot_lolhist(df[df.exp.isin(exp)], 'xlov_ma_abs', filename, xlabel,
                labels, text='No breakup')

    # Plot histogram of overlap velocities (breakup)
    labels = mdfo[mdfo.breakup == 2].label.values
    filename = join(savedirb, expfile +"_hist_lol_breakup_peaks.png")
    plot_lolhist(df[df.exp.isin(exp)], 'xlov_ma_abs_peaks', filename,
                 xlabel, labels, text='Breakup')
    # Plot histogram of overlap velocities (agg)
    labels = mdfo[mdfo.breakup != 2].label.values
    filename = join(savedirb, expfile +"_hist_lol_nobreakup_peaks.png")
    plot_lolhist(df[df.exp.isin(exp)], 'xlov_ma_abs_peaks', filename,
                xlabel,labels, text='No breakup')
    """
