from ._hist import hist_all, hist_aggregating, hist_breakup


__all__ = ['plot_lolhist', 'plot_lolhist_breakup']

def plot_lolhist(df, vcol, filename, xlabel, labels=None, text=None,
                 xscale=None, yscale=None, report=False, fit=False,
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
                         report=False, fit=False,
                         xfitstr=None, yfitstr=None, sigTest=False, cdf=False,
                         legendMean=False, legendTimescale=False, meanL=None,
                         plotMean=False, kde=False):
    """ Saves a histogram of all v in vcol. """
    hist_breakup(df, vcol, filename, xlabel, labelAgg, labels=labels,
                 minTh=None, text=text, xscale=xscale, yscale=yscale,
                 report=report, sigTest=sigTest, cdf=cdf, meanL=meanL,
                 legendMean=legendMean, legendTimescale=legendTimescale,
                 plotMean=plotMean, xfitstr=xfitstr, yfitstr=yfitstr,
                 fit=fit, kde=kde)
