import os.path
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ._hist import hist_aggregating, hist_breakup, hist_peak

def plot_alldhist(overlapDir, df, dflinked, df_tracks, mdfo, meanL, mdfs,
                  exp=None):

    if exp is None:
        exp = np.unique(mdfo.exp.values)

    savedir = join(overlapDir, 'd')
    expfile = ''

    if not os.path.isdir(savedir):
        os.mkdir(savedir)


    D_UNIT = ' [' + u'\u03bc' + 'm]'
    xlabel = r'$d_{glide}$ ' + D_UNIT
    xfitstr = 'd_{glide}'
    input = mdfo[(mdfo.vtot>0) & (~mdfo.vtot.isin([np.nan, np.inf]))]

    filename = join(savedir, expfile +"_hist_dglide_cdf.png")
    hist_aggregating(mdfo[mdfo.exp.isin(exp)], 'vtot', filename, xlabel,
               report=True, cdf=True, fit=True, sigTest=True,
               plotMean=True)
    filename = join(savedir, expfile +"_hist_dglide_cdf_reversals.png")
    hist_peak(mdfo[mdfo.exp.isin(exp)], 'vtot', filename, xlabel,
               report=True, cdf=True, fit=True, sigTest=True,
               plotMean=True)


    slabels = mdfo[mdfo.hasPeaks == 1].label.values
    filename = join(savedir, expfile +"_hist_dglide_cdf_reversals1.png")
    hist_aggregating(input[input.exp.isin(exp)], 'vtot', filename, xlabel, slabels,
               text='Reversing Pairs',
               report=True, cdf=True, fit=True, sigTest=True, plotMean=True)

    slabels = mdfo[mdfo.hasPeaks == 0].label.values
    filename = join(savedir, expfile +"_hist_dglide_cdf_reversals2.png")
    hist_aggregating(input[input.exp.isin(exp)], 'vtot', filename, xlabel, slabels,
               text='Non-reversing Pairs',
               report=True, cdf=True, fit=True, sigTest=True, plotMean=True)




    mdfo['nl'] = mdfo.vtot / (mdfo.length1 + mdfo.length2)
    xlabel = r'$\frac{d_{glide}}{l_1 + l_2}$'
    xfitstr = 'nl'
    input = mdfo[(mdfo.nl>0) & (~mdfo.nl.isin([np.nan, np.inf]))]
    filename = join(savedir, expfile +"_hist_nglide_cdf.png")
    hist_aggregating(input[input.exp.isin(exp)], 'nl', filename, xlabel,
               report=True, cdf=True, fit=True, sigTest=True,
               plotMean=True)
    filename = join(savedir, expfile +"_hist_nglide_cdf_reversals.png")
    hist_peak(input[input.exp.isin(exp)], 'nl', filename, xlabel,
               report=True, cdf=True, fit=True, sigTest=True,
               plotMean=True)

    slabels = mdfo[mdfo.hasPeaks == 1].label.values
    filename = join(savedir, expfile +"_hist_nglide_cdf_reversals1.png")
    hist_aggregating(input[input.exp.isin(exp)], 'nl', filename, xlabel, slabels,
               text='Reversing Pairs',
               report=True, cdf=True, fit=True, sigTest=True, plotMean=True)

    slabels = mdfo[mdfo.hasPeaks == 0].label.values
    filename = join(savedir, expfile +"_hist_nglide_cdf_reversals2.png")
    hist_aggregating(input[input.exp.isin(exp)], 'nl', filename, xlabel, slabels,
               text='Non-reversing Pairs',
               report=True, cdf=True, fit=True, sigTest=True, plotMean=True)


    xlabel = r'$l$'
    xfitstr = 'l'
    agg = mdfo.aggregating.values.tolist()
    exp = mdfo.exp.values.tolist()
    label = mdfo.aggregating.values.tolist()
    l1 = mdfo.length1.values.tolist()
    l2 = mdfo.length2.values.tolist()
    length = pd.DataFrame({'aggregating': agg+agg, 'exp':exp+exp,
                           'label': label+label,
                           'l': l1+l2})
    input = length[(length.l>0) & (~length.l.isin([np.nan, np.inf]))]
    filename = join(savedir, "_hist_length_cdf.png")
    hist_aggregating(input, 'l', filename, xlabel,
                   report=True, cdf=True, legendMean=True, sigTest=True, plotMean=True,
                   xfitstr=xfitstr)

    input = mdfs[(mdfs.length>0) & (mdfs.length<2500) &(~mdfs.length.isin([np.nan, np.inf]))]
    filename = join(savedir, "_hist_length_single_cdf.png")
    hist_aggregating(input, 'length', filename, xlabel,xfitstr=xfitstr,
                   report=True, cdf=True, legendMean=True, sigTest=True, plotMean=True)


    mdfo['ttheor'] = mdfo.nl / (mdfo.v_pos_abs)
    xlabel = r'$t_{theoretisch}$'
    xfitstr = 'tt'
    slabels = mdfo[mdfo.hasPeaks == 1].label.values
    filename = join(savedir, expfile +"_hist_tt_cdf.png")
    hist_aggregating(mdfo[mdfo.exp.isin(exp)], 'ttheor', filename, xlabel,
               report=True, cdf=True, fit=True, sigTest=True,
               plotMean=True)
