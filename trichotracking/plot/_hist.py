import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from scipy.stats import mannwhitneyu, ks_2samp

from ._save import saveplot
from ._fit import fit_semilog
from ._constants import *

from IPython.core.debugger import set_trace


def hist_twoQuantities(df, col, filename, xlabel, cond1, label1, cond2,
                       label2, text=None, xscale=None, yscale=None,
                       xlim=None, cdf=False, kde=False, plotMean=False,
                       bins=None, report=False, fit=False, xfitstr=None,
                       yfitstr=None, legend_out=True, legendMean=False,
                       legendTimescale=False, meanL=None, sigTest=False,
                       c1=None, c2=None, left=False):

    fig, ax = plt.subplots(figsize=FIGSIZE)
    props = {'ax':ax, 'kde':kde, 'norm_hist':True, 'bins':bins}

    if xlim is None:
        xlim = (0.9*df[col].min(), 1.1*df[col].max())
    try:
        ax.set_xlim(xlim)
    except:
        set_trace()

    if cdf:
        props['hist_kws']=dict(cumulative=True)
        props['kde_kws']=dict(cumulative=True)
        props['kde'] = True
        props['hist'] = False
        ylabel = 'CDF'
        if xlim is not None:
            props['kde_kws']['clip']  = xlim
    else:
        ylabel = 'PDF'

    if left:
        posx = 0.2
        posy = 0.41
        ha = 'left'
    else:
        posx = 0.95
        posy = 0.25
        ha = 'right'



    if legendTimescale:
        if meanL is None:
            if 'length' in df:
                dfg = df.groupby('label')
                meanL = dfg.length.mean().mean()
            else:
                meanL = 500
        label1 += r", $\tau = %.2f\,s$" %(meanL/df[cond1][col].mean())
        label2 += r", $\tau = %.2f\,s$" %(meanL/df[cond2][col].mean())

    if (c1 is None) or (c2 is None):
        c1 = cm.tab10(0)
        c2 = cm.tab10(1)

    try:
        sns.distplot(df[cond1][col], label=label1, **props, color=c1)
        if cond2 is not None:
            sns.distplot(df[cond2][col], label=label2, **props, color=c2)
    except:
        set_trace()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)


    ylim = ax.get_ylim()
    ax.set_ylim(ylim)

    if xfitstr is None:
        xfitstr = col

    if fit:
        """fig2, ax2 = plt.subplots(figsize=FIGSIZE)
        props = {'ax':ax2, 'kde':True, 'kde_kws': dict(clip=xlim)}
        sns.distplot(df[cond1][col], label=label1, **props, color=c1)
        sns.distplot(df[cond2][col], label=label2, **props, color=c2)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        for i,c in enumerate((c1,c2)):
            line = ax2.get_lines()[i]
            xd = line.get_xdata()
            yd = line.get_ydata()
            ind = np.argmax(yd)
            coeff = fit_semilog(xd[ind:], yd[ind:])
            e = -int(np.log10(np.abs(coeff[0])))
            polynomial = np.poly1d(coeff)
            xfit = np.linspace(xd[ind], xlim[1], 100)
            yfit = polynomial((xfit))
            ax.semilogy(xfit, np.exp(yfit), ':', lw=1, color=c)"""

        for i, (cond, c) in enumerate(zip([cond1, cond2], [c1, c2])):

            if cond is not None:
                expMean = df[cond][col].mean()
                e = -int(np.log10(np.abs(expMean)))
                textstr = '' if yfitstr is None else yfitstr
                if e < 0:
                    textstr += r" $\propto e^{-\frac{1}{%.0f} %s }$" %(expMean, xfitstr)
                elif np.abs(e) < 2:
                    textstr += r" $\propto e^{-\frac{1}{%.2f} %s }$" %(expMean, xfitstr)
                elif np.abs(e) <3:
                    textstr += r" $\propto e^{-\frac{1}{%.3f} %s }$" %(expMean, xfitstr)
                else:
                    textstr += r" $\propto e^{-\frac{1}{%.4f} %s }$" %(expMean, xfitstr)

                pos = (posx, 0.6-i/15)
                ax.annotate(textstr, xy=pos, xytext=pos, xycoords='figure fraction',
                            color=c, ha=ha)

    elif legendMean:
        for i, (cond, c) in enumerate(zip([cond1, cond2], [c1, c2])):
            if cond is not None:
                textstr = r"$\overline{%s} = %.2f$" %(xfitstr, df[cond][col].mean())
                pos = (posx, 0.6-i/15)
                ax.annotate(textstr, xy=pos, xytext=pos, xycoords='figure fraction',
                            color=c, ha=ha)

    if (cond2 is not None) and (sigTest):
        pos = (posx, posy)
        if df[cond1][col].mean()>df[cond2][col].mean():
            alt='greater'
        else:
            alt='less'
        uvalue, pvalue = mannwhitneyu(df[cond1][col], df[cond2][col], alternative=alt)
        #uvalue, pvalue = ks_2samp(df[cond1][col], df[cond2][col])

        if pvalue < 0.01:
            textstr = r'MWU: $p$-value $<$ 0.01'
        elif pvalue > 0.05:
            textstr = r'MWU: $p$-value $>$ 0.05'
        else:
            textstr = r'MWU: $p$-value = %.2g' %(pvalue)
        ax.annotate(textstr, xy=pos, xytext=pos, xycoords='figure fraction',
                    ha=ha)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plotMean:
        setMeanLine(ax, df[cond1][col].mean(), c1, xlim, 1.06)
        if cond2 is not None:
            setMeanLine(ax, df[cond2][col].mean(), c2, xlim, 1.06)

    legprops = {'frameon': False}
    if text is not None:
        legprops['title'] = text
    if legend_out:
        legprops['loc'] = 'lower center'
        legprops['bbox_to_anchor'] = (0.5, 0.92)

    if cond2 is not None:
        ax.legend(**legprops)

    if report:
        saveplot(fig, filename)

    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')


def setMeanLine(ax, mean, color, xlim, ypos):
    ax.axvline(mean, color=color, linestyle='dashed', linewidth=1)
    mean_figCoords = (mean - xlim[0]) / (xlim[1] - xlim[0])
    #set_trace()
    """ax.text(mean_figCoords, ypos, '{:.2f}'.format(mean),
             transform=ax.transAxes,horizontalalignment='center',
             size='x-small')"""


def getHistData(df, col, labels, minTh, maxTh):
    if labels is None:
        labels = np.unique(df.label.values)

    # Get data points in labels and not nan
    d = df[(df.label.isin(labels))
          &(~df[col].isin([np.inf, -np.inf, np.nan]))]

    if minTh is not None:
        d = d[d[col]>minTh]
    if maxTh is not None:
        d = d[d[col]<maxTh]
    return d

def getLabels(l1, l2, n1, n2):
    if((n1<1000) or (n2<1000)):
        label1 = l1 + r', $n$ = {}'.format(n1)
        label2 = l2 + r', $n$ = {}'.format(n2)
    else:
        label1 = l1
        label2 = l2
    return label1, label2


def hist_all(df, col, filename, xlabel, labels=None, minTh=None,
             maxTh=None, text=None, xscale=None, yscale=None,
             cdf=False, xlim=None, kde=False, plotMean=False,
             bins=None, report=False, fit=False, xfitstr=None,
             yfitstr=None, legend_out=True, appAgglabel='',
             appNaggLabel='', legendMean=False, sigTest=False,
             legendTimescale=False, meanL=None):
    """ Saves a histogram of all v in vcol. """

    d = getHistData(df, col, labels, minTh, maxTh)

    cond1 = (d.aggregating==1)

    c1 = AGG_COLOR
    c2 = NAGG_COLOR

    hist_twoQuantities(d, col, filename, xlabel, cond1, '', None,
                           None, text=text, xscale=xscale, yscale=yscale,
                           cdf=cdf, xlim=xlim, kde=kde, plotMean=plotMean,
                           bins=bins, report=report, fit=fit,
                           xfitstr=xfitstr, yfitstr=yfitstr,
                           legend_out=legend_out, legendMean=legendMean,
                           legendTimescale=legendTimescale, sigTest=sigTest,
                           meanL=meanL, c1=c1, c2=c2)



def hist_aggregating(df, col, filename, xlabel, labels=None, minTh=None,
                     maxTh=None, text=None, xscale=None, yscale=None,
                     cdf=False, xlim=None, kde=False, plotMean=False,
                     bins=None, report=False, fit=False, xfitstr=None,
                     yfitstr=None, legend_out=True, appAgglabel='',
                     appNaggLabel='', legendMean=False, sigTest=False,
                     legendTimescale=False, meanL=None, left=False):
    """ Saves a histogram of all v in vcol. """

    d = getHistData(df, col, labels, minTh, maxTh)

    cond1 = (d.aggregating==1)
    cond2 = (d.aggregating==0)

    label1, label2 = getLabels('Aggregating', 'Non-aggregating',
                                d[cond1][col].size, d[cond2][col].size)

    label1 += appAgglabel
    label2 += appNaggLabel

    c1 = AGG_COLOR
    c2 = NAGG_COLOR

    hist_twoQuantities(d, col, filename, xlabel, cond1, label1, cond2,
                           label2, text=text, xscale=xscale, yscale=yscale,
                           cdf=cdf, xlim=xlim, kde=kde, plotMean=plotMean,
                           bins=bins, report=report, fit=fit,
                           xfitstr=xfitstr, yfitstr=yfitstr,
                           legend_out=legend_out, legendMean=legendMean,
                           legendTimescale=legendTimescale, sigTest=sigTest,
                           meanL=meanL, c1=c1, c2=c2, left=left)


def hist_breakup(df, col, filename, xlabel, labelAgg, labels=None,
                     minTh=None, maxTh=None, text=None, xscale=None,
                     yscale=None, cdf=False, xlim=None, kde=False,
                     plotMean=False, bins=None, report=False,fit=False,
                     xfitstr=None, yfitstr=None, legend_out=True,
                     sigTest=False, legendTimescale=False, meanL=None,
                     legendMean=False):

    d = getHistData(df, col, labels, minTh, maxTh)

    cond1 = (d.breakup!=2)
    cond2 = (d.breakup==2)

    if labelAgg != '':
        l1 = labelAgg + ', No separation'
        l2 = labelAgg + ', Separation'
    else:
        l1 = 'No separation'
        l2 = 'Separation'

    c1 = NSPLIT_COLOR
    c2 = SPLIT_COLOR

    label1, label2 = getLabels(l1,l2, d[cond1][col].size, d[cond2][col].size)

    hist_twoQuantities(d, col, filename, xlabel, cond1, label1, cond2,
                           label2, text=text, xscale=xscale, yscale=yscale,
                           cdf=cdf, xlim=xlim, kde=kde, plotMean=plotMean,
                           bins=bins, report=report, fit=fit, xfitstr=xfitstr,
                           yfitstr=yfitstr, legend_out=legend_out,
                           sigTest=sigTest, legendMean=legendMean,
                           legendTimescale=legendTimescale, meanL=meanL,
                           c1=c1, c2=c2)



def hist_lol(df, col, filename, xlabel, labelAgg, labels=None, minTh=None,
             maxTh=None, text=None, xscale=None, yscale=None, cdf=False,
             xlim=None, kde=False, plotMean=False, bins=None, report=False,
             fit=False, xfitstr=None, yfitstr=None, legend_out=True,
             sigTest=False, legendTimescale=False, meanL=None,
             legendMean=False):

    d = getHistData(df, col, labels, minTh, maxTh)

    cond1 = ((d.xlov_ma_abs<0.01) & (d.ylov.abs()<0.01))
    cond2 = ((d.xlov_ma_abs>0.01) | (d.ylov.abs()>0.01))

    label1 = labelAgg+ r', LOL = 0, $n$ = {}'.format(d[cond1][col].size)
    label2 = labelAgg + r', LOL > 0 $n$ = {}'.format(d[cond2][col].size)
    hist_twoQuantities(d, col, filename, xlabel, cond1, label1, cond2,
                           label2, text=text, xscale=xscale, yscale=yscale,
                           cdf=cdf, xlim=xlim, kde=kde, plotMean=plotMean,
                           bins=bins, report=report, fit=fit, xfitstr=xfitstr,
                           yfitstr=yfitstr, legend_out=legend_out,
                           sigTest=sigTest, legendMean=legendMean,
                           legendTimescale=legendTimescale)

def hist_peak(df, col, filename, xlabel, labels=None, minTh=None,
             maxTh=None, text=None, xscale=None, yscale=None, cdf=False,
             xlim=None, kde=False, plotMean=False, bins=None, report=False,
             fit=False, xfitstr=None, yfitstr=None, legend_out=True,
             sigTest=False, legendTimescale=False, meanL=None,
             legendMean=False):

    d = getHistData(df, col, labels, minTh, maxTh)

    cond1 = (d.hasPeaks)
    cond2 = (~d.hasPeaks)

    c1 = REV_COLOR
    c2 = NREV_COLOR

    label1, label2 = getLabels('Reversing', 'Non-reversing',
                                d[cond1][col].size, d[cond2][col].size)
    hist_twoQuantities(d, col, filename, xlabel, cond1, label1, cond2,
                           label2, text=text, xscale=xscale, yscale=yscale,
                           cdf=cdf, xlim=xlim, kde=kde, plotMean=plotMean,
                           bins=bins, report=report, fit=fit, xfitstr=xfitstr,
                           yfitstr=yfitstr, legend_out=legend_out,
                           sigTest=sigTest, legendMean=legendMean,
                           legendTimescale=legendTimescale, c1=c1,
                           c2=c2)
