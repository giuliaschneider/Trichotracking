import pandas as pd
import seaborn as sns
from IPython.terminal.debugger import set_trace
from trichotracking.dfmanip import fit_log

from ._constants import *
from ._save import saveplot

__all__ = ['scatterplot']


def plot_scatterTrueFalse(df, xcol, ycol, huecol, labelTrue, labelFalse,
                          ctrue, cfalse):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    dftrue = df[df[huecol]]
    dffalse = df[~df[huecol]]
    ax.scatter(dftrue[xcol], dftrue[ycol], label=labelTrue, color=ctrue)
    ax.scatter(dffalse[xcol], dffalse[ycol], label=labelFalse, color=cfalse)
    ax.legend()
    return fig, ax


def plot_scatter_multiple(df, xcol, ycol, conds, labels, colors):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)

    for cond, label, c in zip(conds, labels, colors):
        dfcond = df[cond]
        ax.scatter(dfcond[xcol], dfcond[ycol], label=label, color=c)
    ax.legend()
    return fig, ax


def scatterplot(df, xcol, ycol, huecol, filename, xlabel, ylabel,
                xscale=None, yscale=None, ylim=None, xlim=None, fit=False,
                xfitstr=None, yfitstr=None, report=False,
                separate_fit=False, legend_title=None, legend_out=False,
                m_quantile=False):
    # fig, ax = plt.subplots(figsize=(4.3, 3.3))
    # props = {'palette': 'ch:_r'}
    if huecol == 'hasPeaks':
        l1 = 'Reversing'
        l2 = 'Non-reversing'
        c1 = REV_COLOR
        c2 = NREV_COLOR
        colors = [c1, c2]

        fig, ax = plot_scatterTrueFalse(df, xcol, ycol, huecol, l1, l2, c1, c2)
    elif huecol == 'aggregating':
        l1 = 'Menadione'
        l2 = 'Control'
        c1 = AGG_COLOR
        c2 = NAGG_COLOR
        fig, ax = plot_scatterTrueFalse(df, xcol, ycol, huecol, l1, l2, c1, c2)
        colors = [c1, c2]

    elif huecol == 'breakup':
        d = df.copy()
        d['bb'] = False
        d.loc[d.breakup != 2, 'bb'] = True
        l1 = 'No split'
        l2 = 'Split'
        c1 = NSPLIT_COLOR
        c2 = SPLIT_COLOR
        fig, ax = plot_scatterTrueFalse(d, xcol, ycol, 'bb', l1, l2, c1, c2)
    elif huecol == 'revb':
        conds = [(df.revb == 1), (df.revb == 2), (df.revb == 4)]
        labels = REV_SPLIT_LABEL_W
        colors = REV_SPLIT_COLORS_W
        fig, ax = plot_scatter_multiple(df, xcol, ycol, conds, labels, colors)

    else:
        props = {'scatter_kws': {"s": MARKERSIZE ** 2}}
        ax = sns.scatterplot(x=df[xcol], y=df[ycol], hue=df[huecol])
        plt.gcf().set_size_inches(FIGSIZE)

    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xlim is None:
        x = df[xcol]
        xlim = (0.9 * x.min(), 1.1 * x.max())
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if fit:
        coeff = fit_log(df, xcol, ycol)
        polynomial = np.poly1d(coeff)
        xfit = np.linspace(xlim[0], xlim[1], 100)
        yfit = polynomial(np.log(xfit))
        ax.loglog(xfit, np.exp(yfit), 'k:', lw=1)
        textstr = ycol if yfitstr is None else yfitstr
        textstr += r" $\propto$ "
        textstr += xcol if xfitstr is None else xfitstr
        textstr += r"$^{%.2f}$" % coeff[0]
        pos = (0.2, 0.9)
        ax.annotate(textstr, xy=pos, xytext=pos, xycoords='figure fraction',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              alpha=0.3, ec='white'))

    if separate_fit:
        items = np.unique(df[huecol])
        if items.dtype == np.dtype('bool'):
            items = items[::-1]
        for i, (item, c) in enumerate(zip(items, colors)):
            dfitem = df[df[huecol] == item]
            coeff = fit_log(dfitem, xcol, ycol)
            polynomial = np.poly1d(coeff)
            xfit = np.linspace(xlim[0], xlim[1], 100)
            yfit = polynomial(np.log(xfit))
            ax.loglog(xfit, np.exp(yfit), ':', lw=1, color=c)
            textstr = ycol if yfitstr is None else yfitstr
            textstr += r" $\propto$ "
            textstr += xcol if xfitstr is None else xfitstr
            textstr += r"$^{%.2f}$" % coeff[0]
            if legend_out:
                pos = (0.21, 0.7 - i / 14)
            else:
                pos = (0.21, 0.90 - i / 14)

            ax.annotate(textstr, xy=pos, xytext=pos, xycoords='figure fraction',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  alpha=0.3, ec='white'), color=c)

    if m_quantile:
        n = df[xcol].size
        delta = xlim[1] - xlim[0]
        bins = np.linspace(xlim[0], xlim[1], int(n / delta / 5))

        groups = df.groupby(pd.cut(df[xcol], bins))
        xmean = groups[xcol].mean()
        yquantile = groups[ycol].quantile(0.95)

        ax.loglog(xmean, yquantile, 'k:')

    legend = ax.get_legend()
    legend.get_frame().set_edgecolor('none')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    bb = legend.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    # Change to location of the legend.
    if legend_out:
        bb.x0 = -0.1
        bb.y0 = 0.95
    else:
        bb.x0 = 0.35
        bb.y0 = 0.95
    legend.set_bbox_to_anchor(bb, )

    # legend.set_loc('lower center')
    if legend_title is not None:
        legend.set_title(legend_title)

    if report:
        saveplot(plt.gcf(), filename)
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')
