# -*- coding: utf-8 -*-

import os.path
from os.path import join
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns

from ._hist import hist_aggregating, hist_breakup, hist_lol,hist_peak, getHistData
from ._save import saveplot


V_UNIT = ' [' + u'\u03bc' + 'm/s]'

def plot_vhist(df, vcol, filename, xlabel, labels=None, text=None,
               xscale=None, yscale=None, cdf=False, maxTh=4, minTh=None,
               xlim=None, report=False, legendMean=False, xfitstr=None,
               yfitstr=None, kde=False, legend_out=True, legendfit=False,
               appAgglabel='', appNaggLabel='', legendTimescale=False,
               meanL=None, sigTest=False, plotMean=False):
    """ Saves a histogram of all v in vcol. """
    hist_aggregating(df, vcol, filename, xlabel, labels=labels, maxTh=maxTh,
                    text=text, xscale=xscale, yscale=yscale, cdf=cdf,
                    minTh=minTh, xlim=xlim, report=report,
                    xfitstr=xfitstr, yfitstr=yfitstr, kde=kde,
                    legend_out=legend_out, legendMean=legendMean,
                    appAgglabel=appAgglabel, appNaggLabel=appNaggLabel,
                    legendTimescale=legendTimescale, meanL=meanL,
                    sigTest=sigTest, plotMean=plotMean)

def plot_vhist_breakup(df, vcol, filename, xlabel, labels=None, text=None,
                   labelAgg='Aggregating', xscale=None, yscale=None,
                   cdf=False, report=False, legendMean=False, xfitstr=None,
                   yfitstr=None, kde=False, legend_out=True, sigTest=False,
                   legendfit=False, legendTimescale=False, plotMean=False):
    """ Saves a histogram of all v in vcol. """
    hist_breakup(df, vcol, filename, xlabel, labelAgg,labels=labels,
                 maxTh=4, text=text, xscale=xscale, yscale=yscale,
                 cdf=cdf, report=report, xfitstr=xfitstr,
                 yfitstr=yfitstr, kde=kde, legend_out=legend_out,
                 sigTest=sigTest,legendMean=legendMean,
                 legendTimescale=legendTimescale, plotMean=plotMean)


def plot_vhist_lol(df, vcol, filename, xlabel, labels=None, text=None,
                   labelAgg='Aggregating', xscale=None, yscale=None,
                   cdf=False, report=False, legendfit=False, xfitstr=None,
                   yfitstr=None, kde=False, legend_out=True, sigTest=False,
                   legendMean=False, legendTimescale=False, plotMean=False):
    """ Saves a histogram of all v in vcol. """
    hist_lol(df, vcol, filename, xlabel, labelAgg,labels=labels,
                 maxTh=4, text=text, xscale=xscale, yscale=yscale,
                 cdf=cdf, report=report,  xfitstr=xfitstr,
                 yfitstr=yfitstr, kde=kde, legend_out=legend_out,
                 sigTest=sigTest,legendMean=legendMean,
                 legendTimescale=legendTimescale, plotMean=plotMean)


def plot_vhist_peak(df, vcol, filename, xlabel, labels=None, text=None,
                   labelAgg='Aggregating', xscale=None, yscale=None,
                   cdf=False, report=False, legendMean=False, xfitstr=None,
                   yfitstr=None, kde=False, legend_out=True, sigTest=False,
                   legendfit=False, legendTimescale=False, plotMean=False):
    """ Saves a histogram of all v in vcol. """
    hist_peak(df, vcol, filename, xlabel, labels=labels, minTh=-4,
                 maxTh=4, text=text, xscale=xscale, yscale=yscale,
                 cdf=cdf, report=report,  xfitstr=xfitstr,
                 yfitstr=yfitstr, kde=kde, legend_out=legend_out,
                 sigTest=sigTest, legendMean=legendMean,
                 legendTimescale=legendTimescale, plotMean=plotMean)




def plot_allvhist(overlapDir, df, dflinked, df_tracks, mdfo, mdfs, meanL,
                  exp=None):

    if exp is None:
        exp = np.unique(mdfo.exp.values)

    if isinstance(exp, str):
        savedirexp = join(overlapDir, exp)
        if not os.path.isdir(savedirexp):
            os.mkdir(savedirexp)
        savedir = os.path.join(overlapDir, exp, 'v')
        expfile = exp
        exp = [exp]
    else:
        savedir = join(overlapDir, 'v')
        expfile = ''

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    savedirSingle = join(savedir, 'vsingle')
    if not os.path.isdir(savedirSingle):
        os.mkdir(savedirSingle)

    savedirb = join(savedir, 'overlap_breakup_nobreakup')
    if not os.path.isdir(savedirb):
        os.mkdir(savedirb)

    savediragg = join(savedir, 'overlap_aggregating_nonaggregating')
    if not os.path.isdir(savediragg):
        os.mkdir(savediragg)

    savedirlol = join(savedir, 'overlap_aggregating_nonaggregating')
    if not os.path.isdir(savedirlol):
        os.mkdir(savedirlol)



    # vsingle
    ####################################################################

    xlabel = r'$|v_{s}|$'+ V_UNIT
    xfitstr= r'|v_s|'
    slabels = df_tracks[df_tracks.type==1].label
    # Plot histogram of absolute single filament velocities
    filename = join(savedirSingle, expfile+"_hist_vsingle_cdf.png")
    plot_vhist(mdfs[mdfs.exp.isin(exp)], 'v_abs', filename, xlabel,
              cdf=True, report=True, legend_out=True,
              legendMean=True, meanL=meanL,
              sigTest=True, xfitstr=xfitstr)
    plot_vhist(mdfs[mdfs.exp.isin(exp)], 'v_abs', filename, xlabel,
              cdf=True, report=True, legend_out=True,
              legendMean=True, meanL=meanL,
              sigTest=True, xfitstr=xfitstr)
    """
    filename = join(savedirSingle, expfile+"_hist_vsingle.png")
    plot_vhist(mdfs[mdfs.exp.isin(exp)], 'v_abs', filename, xlabel,
               slabels)
    # Plot histogram of absolute single filament velocities


    # Plot histogram of absolute single filament velocities
    filename = join(savedirSingle, expfile+"_hist_vsingle_loglog_cdf.png")
    plot_vhist(mdfs[mdfs.exp.isin(exp)], 'v_abs', filename, xlabel,
               slabels, xscale='log', yscale='log', cdf=True)
    # Plot histogram of absolute single filament velocities
    filename = join(savedirSingle, expfile+"_hist_vsingle_semilog.png")
    plot_vhist(mdfs[mdfs.exp.isin(exp)], 'v_abs', filename, xlabel,
               slabels, yscale='log')
    filename = join(savedirSingle, expfile+"_hist_vsingle_semilog_cdf.png")
    plot_vhist(mdfs[mdfs.exp.isin(exp)], 'v_abs', filename, xlabel,
               slabels, yscale='log', cdf=True)"""

    # vmean / (rt - r0) / t
    ####################################################################
    """
    xlabel = r'$\frac{\bar{|v|}}{|r_T - r_0|/T}$' + V_UNIT
    slabels = df_tracks[df_tracks.type==1].label
    filename = join(savedirSingle, expfile+"_hist_vdist.png")
    input = df_tracks[(df_tracks.exp.isin(exp)) & ( df_tracks.v_dist>=1)]
    plot_vhist(input, 'v_dist', filename, xlabel,
               slabels, xlim=(1,20))
    filename = join(savedirSingle, expfile+"_hist_vdist_cdf.png")
    plot_vhist(input, 'v_dist', filename, xlabel,
               slabels, maxTh=None, cdf=True, xlim=(1,4))
    filename = join(savedirSingle, expfile+"_hist_vdist_loglog.png")
    plot_vhist(input, 'v_dist', filename, xlabel, slabels,
               xlim=(1,50), xscale='log', yscale='log', maxTh=None)
    filename = join(savedirSingle, expfile+"_hist_vdist_loglog_cdf.png")
    plot_vhist(input, 'v_dist', filename, xlabel, slabels,
               xlim=(1,50), maxTh=None, cdf=True, xscale='log', yscale='log')
    """



    # v overlap
    ####################################################################
    # Plot histogram of overlap velocities
    xlabel = r'$|v_{r}|$' + V_UNIT
    xfitstr = r'|v_{r}|'
    yfitstr = ''
    """
    filename = join(savedir, expfile +"_hist_voverlap.png")
    plot_vhist(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel,
               report=True)
    filename = join(savedir, expfile +"_hist_voverlap_loglog.png")
    plot_vhist(df[df.exp.isin(exp)], 'v_pos_abs', filename, xlabel,
               xscale='log', yscale='log')

    filename = join(savedir, expfile +"_hist_voverlap_semilog.png")
    plot_vhist(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel,
               yscale='log')
    """
    filename = join(savedir, expfile +"_hist_voverlap_cdf.png")
    plot_vhist(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel, cdf=True,
               report=True, legend_out=True, sigTest=True, plotMean=True,
               legendMean=True, xfitstr=xfitstr)



    # v overlap: agg / non aggregating cultures: breakup no breakup pairs
    ####################################################################
    """
    labels = mdfo[mdfo.breakup == 2].label.values
    filename = join(savedirb, expfile +"_hist_voverlap_breakup.png")
    plot_vhist(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel, labels, text='Breakup')
    # Plot histogram of overlap velocities (agg)
    labels = mdfo[mdfo.breakup != 2].label.values
    filename = join(savedirb, expfile +"_hist_voverlap_nobreakup.png")
    plot_vhist(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel,labels, text='Agg')

    # v overlap: aggregating breakup / no breakup
    ####################################################################
    # Aggregating, breakup vs no breakup
    labels = mdfo[mdfo.aggregating == 1].label.values
    filename = join(savediragg, expfile +"_hist_voverlap_aggregating.png")
    plot_vhist_breakup(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel,labels)
    # Non aggregating
    labels = mdfo[mdfo.aggregating == 0].label.values
    filename = join(savediragg, expfile +"_hist_voverlap_nonaggregating.png")
    plot_vhist_breakup(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel,labels,
                   labelAgg='Non aggregating')


    filename = join(savedirb, expfile +"_hist_voverlap_all_semilog.png")
    plot_vhist_breakup(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel,
                      labels=None, labelAgg='', yscale='log', report=True,
                      legendMean=True, xfitstr=xfitstr, yfitstr=yfitstr)
    """
    filename = join(savedirb, expfile +"_hist_voverlap_split_cdf.png")
    plot_vhist_breakup(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel,
                      labels=None, labelAgg='', report=True, cdf=True,
                      legend_out=True, sigTest=True, plotMean=True,
                      legendMean=True, xfitstr=xfitstr)
    filename = join(savedirb, expfile +"_hist_voverlap_split.png")
    plot_vhist_breakup(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel,
                      labels=None, labelAgg='', report=True, kde=True,
                      legend_out=True, sigTest=True, plotMean=True,
                      legendMean=True, xfitstr=xfitstr)

    # v overlap: peaks
    ####################################################################
    """
    filename = join(savedir, expfile +"_hist_voverlap_peaks.png")
    plot_vhist_peak(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel,
                    report=True)
    filename = join(savedir, expfile +"_hist_voverlap_peaks_semilog.png")
    plot_vhist_peak(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel,
                    yscale='log', report=True, legendMean=True, xfitstr=xfitstr,
                    yfitstr=yfitstr)
    """
    filename = join(savedir, expfile +"_hist_voverlap_peaks_cdf.png")
    plot_vhist_peak(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel,
               cdf=True, report=True, legend_out=True,
               sigTest=True, plotMean=True, legendMean=True, xfitstr=xfitstr)

    filename = join(savedir, expfile +"_hist_voverlap_reversals1_cdf.png")
    slabels = mdfo[(mdfo.hasPeaks == 1)].label.values
    plot_vhist(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel, slabels,
               text='Reversing Pairs', report=True, cdf=True, legendMean=True,
               sigTest=True, plotMean=True)

    slabels = mdfo[(mdfo.hasPeaks == 0)].label.values
    filename = join(savedir, expfile +"_hist_voverlap_reversals2_cdf.png")
    plot_vhist(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel, slabels,
               text='Non-reversing Pairs', report=True, legendMean=True,
               xfitstr=xfitstr, yfitstr=yfitstr, sigTest=True, cdf=True,
               plotMean=True)
    slabels = mdfo[(mdfo.revb == 1)].label.values
    filename = join(savedir, expfile +"_hist_voverlap_reversals3_cdf.png")
    plot_vhist(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel, slabels,
               text='Reversing, non-separating pairs', report=True, legendMean=True,
               xfitstr=xfitstr, yfitstr=yfitstr, sigTest=True, cdf=True,
               plotMean=True)

    slabels = mdfo[(mdfo.revb == 2)].label.values
    filename = join(savedir, expfile +"_hist_voverlap_reversals4_cdf.png")
    plot_vhist(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel, slabels,
               text='Reversing, separating pairs', report=True, legendMean=True,
               xfitstr=xfitstr, yfitstr=yfitstr, sigTest=True, cdf=True,
               plotMean=True)

    slabels = mdfo[(mdfo.revb == 4)].label.values
    filename = join(savedir, expfile +"_hist_voverlap_reversals5_cdf.png")
    plot_vhist(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel, slabels,
               text='Non-reversing, separating pairs', report=True, legendMean=True,
               xfitstr=xfitstr, yfitstr=yfitstr, sigTest=True, cdf=True,
               plotMean=True)
    slabels = mdfo[(mdfo.revb == 3)].label.values
    filename = join(savedir, expfile +"_hist_voverlap_reversals6_cdf.png")
    plot_vhist(mdfo[mdfo.exp.isin(exp)], 'v_pos_abs', filename, xlabel, slabels,
               text='Non-reversing, non-separating pairs', report=True, legendMean=True,
               xfitstr=xfitstr, yfitstr=yfitstr, sigTest=True, cdf=True,
               plotMean=True)

    # v lol
    ####################################################################
    """xlabel = r'$|v_{lol}|$' + V_UNIT
    filename = join(savedir, expfile +"_hist_vlol.png")
    plot_vhist(df[df.exp.isin(exp)], 'vlol', filename, xlabel, maxTh=3,
               minTh=-3)
    filename = join(savedir, expfile +"_hist_vlol_semilog.png")
    plot_vhist(df[df.exp.isin(exp)], 'vlol', filename, xlabel, maxTh=30,
               minTh=-30, yscale='log')
    filename = join(savedir, expfile +"_hist_vlol_peaks.png")
    input = df[ (df.xlov_ma_abs>0.01) & (df.exp.isin(exp)) ]
    plot_vhist_peak(input, 'vlol', filename, xlabel)
    """
    # v overlap: aggregating lol / no lol
    ####################################################################

    xlabel = r'$|v_{overlap}|$' + V_UNIT
    filename = join(savediragg, expfile +"_hist_voverlap_lol0.png")
    plot_vhist(mdfo[mdfo.exp.isin(exp)], 'v_pos_abslol0', filename, xlabel,
                   text=r'$LOL$ = 0', report=True, legendMean=True,
                   xfitstr=xfitstr, yfitstr=yfitstr, sigTest=True, cdf=True,
                   plotMean=True)
    # Non aggregating, LOL=0 vs LOL > 0
    filename = join(savediragg, expfile +"_hist_voverlap_lol1.png")
    plot_vhist(mdfo[mdfo.exp.isin(exp)], 'v_pos_abslol1', filename, xlabel,
               text=r'$LOL>0$', report=True, legendMean=True,
               xfitstr=xfitstr, yfitstr=yfitstr, sigTest=True, cdf=True,
               plotMean=True)

    filename = join(savediragg, expfile +"_hist_voverlap_lol2.png")
    plot_vhist(mdfo[mdfo.exp.isin(exp)], 'v_pos_abslol2', filename, xlabel,
               text='Reversing, separating pairs', report=True, legendMean=True,
               xfitstr=xfitstr, yfitstr=yfitstr, sigTest=True, cdf=True,
               plotMean=True)


    # v single: single or in pairs
    ####################################################################
    """
    xlabel = r'$|v|$ ' + V_UNIT
    filename = join(savedirSingle, expfile +"_hist_vsingle_aggregating.png")
    plot_vhist_single(df, dflinked, df_tracks,filename, xlabel, agg=True, text=None,
                       labelAgg='Aggregating', report=True)
    filename = join(savedirSingle, expfile +"_hist_vsingle_aggregating_semilog.png")
    plot_vhist_single(df, dflinked, df_tracks,filename, xlabel, agg=True, text=None,
                       labelAgg='Aggregating', report=True, yscale='log')
    # Non aggregating
    filename = join(savedirSingle, expfile +"_hist_vsingle_nonaggregating.png")
    plot_vhist_single(df, dflinked, df_tracks,filename, xlabel, agg=False, text=None,
                       labelAgg='Non aggregating', report=True)
    filename = join(savedirSingle, expfile +"_hist_vsingle_nonaggregating_semilog.png")
    plot_vhist_single(df, dflinked, df_tracks,filename, xlabel, agg=False, text=None,
                       labelAgg='Non aggregating', report=True,yscale='log')
    """






def plot_vhist_single(df, dflinked, df_tracks, filename, xlabel, agg=True, text=None,
                   labelAgg='Aggregating', report=False, yscale=None):
    """ Saves a histogram of all v in vcol. """

    fig, ax = plt.subplots()
    props = {'ax':ax, 'kde':False, 'norm_hist':True}
    if agg:
        labelS = df_tracks[(df_tracks.type==1) & (df_tracks.aggregating==1)].label.values
        labelF = df[(df.aggregating==1)].label.values
    else:
        labelS = df_tracks[(df_tracks.type==1) & (df_tracks.aggregating==0)].label.values
        labelF = df[(df.aggregating==0)].label.values

    d = getHistData(dflinked, 'v_abs', labelS, None, maxTh=4).v_abs

    e = getHistData(df, 'v1', labelF, minTh=-4,  maxTh=4).v1.abs().values.tolist()
    e += getHistData(df, 'v2', labelF, minTh=-4, maxTh=4).v2.abs().values.tolist()
    f = pd.DataFrame({'v_abs':e}).v_abs
    nagg = d.size
    labelagg = labelAgg + r', single, $n$ = {}'.format(nagg)
    nnagg = f.size
    labelnagg = labelAgg + r', single in pairs $n$ = {}'.format(nnagg)
    sns.distplot(d, label=labelagg, **props)
    sns.distplot(f, label=labelnagg, **props)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_xlim([0, 4])
    if yscale is not None:
        ax.set_yscale(yscale)
    if text is not None:
        ax.legend(frameon=False, title=text)
    else:
        ax.legend(frameon=False)
    if report:
        saveplot(fig, filename)
    fig.savefig(filename)
    plt.close(fig)
