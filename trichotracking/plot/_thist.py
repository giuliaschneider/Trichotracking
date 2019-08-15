from ._hist import hist_aggregating, hist_breakup

__all__ = ['plot_thist', 'plot_thist_breakup']


def plot_thist(df, vcol, filename, xlabel, labels=None, text=None,
               xscale=None, yscale=None, report=False, fit=False,
               xfitstr=None, yfitstr=None, sigTest=False, cdf=False,
               legendMean=False, legendTimescale=False, meanL=None,
               plotMean=False, xlim=None):
    """ Saves a histogram of all v in vcol. """
    hist_aggregating(df, vcol, filename, xlabel, labels=labels,
                     text=text, xscale=xscale, yscale=yscale, report=report,
                     fit=fit, xfitstr=xfitstr, yfitstr=yfitstr,
                     minTh=1, sigTest=sigTest, cdf=cdf, meanL=meanL,
                     legendMean=legendMean, legendTimescale=legendTimescale,
                     plotMean=plotMean, xlim=xlim)


def plot_thist_breakup(df, vcol, filename, xlabel, labels=None, text=None,
                       labelAgg='Aggregating', xscale=None, yscale=None,
                       report=False, fit=False, xfitstr=None, yfitstr=None,
                       sigTest=False, legendMean=False, legendTimescale=False,
                       meanL=None, cdf=False, plotMean=False, kde=False,
                       xlim=None):
    """ Saves a histogram of all v in vcol. """
    hist_breakup(df, vcol, filename, xlabel, labelAgg, labels=labels,
                 text=text, xscale=xscale, yscale=yscale, report=report,
                 fit=fit, xfitstr=xfitstr, yfitstr=yfitstr,
                 sigTest=sigTest, cdf=cdf, meanL=meanL,
                 legendMean=legendMean, legendTimescale=legendTimescale,
                 plotMean=plotMean, kde=kde, xlim=xlim)
