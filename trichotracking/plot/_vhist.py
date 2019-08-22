from ._hist import hist_aggregating

__all__ = ['plot_vhist']


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
