import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from ._save import saveplot
from ._fit import fit_log
from ._constants import *

from IPython.core.debugger import set_trace



def plot_scatterTrueFalse(df, xcol, ycol, huecol, labelTrue, labelFalse,
                          ctrue, cfalse):
    fig = plt.figure(figsize=FIGSIZE)
    ax = ax = fig.add_subplot(111)
    dftrue = df[df[huecol]==True]
    dffalse = df[df[huecol]==False]
    ax.scatter(dftrue[xcol], dftrue[ycol], label=labelTrue, color=ctrue)
    ax.scatter(dffalse[xcol], dffalse[ycol], label=labelFalse, color=cfalse)
    ax.legend()
    return fig, ax

def plot_scatter_multiple(df, xcol, ycol, conds, labels, colors):
    fig = plt.figure(figsize=FIGSIZE)
    ax = ax = fig.add_subplot(111)

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

    #fig, ax = plt.subplots(figsize=(4.3, 3.3))
    #props = {'palette': 'ch:_r'}
    if huecol == 'hasPeaks':
        l1 = 'Reversing'
        l2 = 'Non-reversing'
        c1 = REV_COLOR
        c2 = NREV_COLOR
        colors = [c1, c2]

        fig, ax = plot_scatterTrueFalse(df, xcol, ycol, huecol, l1, l2, c1, c2)
    elif huecol == 'aggregating':
        l1 = 'Aggregating'
        l2 = 'Non-aggregating'
        c1 = AGG_COLOR
        c2 = NAGG_COLOR
        fig, ax = plot_scatterTrueFalse(df, xcol, ycol, huecol, l1, l2, c1, c2)
        colors = [c1, c2]

    elif huecol == 'breakup':
        d = df.copy()
        d['bb'] = False
        d.loc[d.breakup!=2, 'bb'] = True
        l1 = 'No split'
        l2 = 'Split'
        c1 = NSPLIT_COLOR
        c2 = SPLIT_COLOR
        fig, ax = plot_scatterTrueFalse(d, xcol, ycol, 'bb', l1, l2, c1, c2)
    elif huecol == 'revb':
        conds = [(df.revb==1), (df.revb==2), (df.revb==4)]
        labels = REV_SPLIT_LABEL_W
        colors = REV_SPLIT_COLORS_W
        fig, ax = plot_scatter_multiple(df, xcol, ycol, conds, labels, colors)

    else:
        props = {'scatter_kws': {"s": MARKERSIZE**2}}
        ax = sns.scatterplot(x=df[xcol], y=df[ycol], hue=df[huecol])
        plt.gcf().set_size_inches(FIGSIZE)

    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xlim is None:
        x = df[xcol]
        xlim = (0.9*x.min(), 1.1*x.max())
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
        textstr += r"$^{%.2f}$" %coeff[0]
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
            textstr += r"$^{%.2f}$" %coeff[0]
            if legend_out:
                pos = (0.21, 0.7-i/14)
            else:
                pos = (0.21, 0.90-i/14)

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
    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    bb = legend.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    # Change to location of the legend.
    if legend_out:
        bb.x0 = -0.1
        bb.y0 = 0.95
    else:
        bb.x0 = 0.35
        bb.y0 = 0.95
    legend.set_bbox_to_anchor(bb,)

    #legend.set_loc('lower center')
    if legend_title is not None:
        legend.set_title(legend_title)


    if report:
        saveplot(plt.gcf(), filename)
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')





def v_t(overlapDir, df, dflinked, df_tracks, mdfo, exp=None):

    mf = mdfo[mdfo.revb!=3]

    if exp is None:
        exp = np.unique(mf.exp.values)

    if isinstance(exp, str):
        savedirexp = os.path.join(overlapDir, exp)
        if not os.path.isdir(savedirexp):
            os.mkdir(savedirexp)
        savedir = os.path.join(overlapDir, exp, 'scatter')
        expfile = exp
        exp = [exp]
    else:
        savedir = os.path.join(overlapDir, 'scatter')
        expfile = ''

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    """
    # v vs t
    ####################################################################
    xlabel = r'$t$ [s]'
    ylabel = r'$v_r$ [µm/s]'
    input = mf[(~mf.t.isnull()) & (mf.t>0) & (~mf.vlol.isin([np.inf]))]

    filename=os.path.join(savedir, 'v_t.png')
    scatterplot(input, 't', 'v_pos_abs', 'peaks',filename,xlabel,ylabel,
                ylim=(0,4), report=True)

    filename=os.path.join(savedir, 'v_t_loglog.png')
    scatterplot(input, 't', 'vlol', 'aggregating', filename, xlabel,
                ylabel, xscale='log', yscale='log', fit=True,
                yfitstr=r'$v_r$', report=True)


    input = mf[(~mf.t.isnull()) & (mf.t>0) & (mf.peaks<1)]
    filename=os.path.join(savedir, 'v_t_loglog_peaks0.png')
    scatterplot(input, 't', 'v_pos_abs', 'peaks', filename, xlabel,
                ylabel, xscale='log', yscale='log', fit=True,
                yfitstr=r'$v_r$', report=True)
    filename=os.path.join(savedir, 'v_t_loglog_peaks0_agg.png')
    scatterplot(input, 't', 'v_pos_abs', 'aggregating', filename, xlabel,
                ylabel, xscale='log', yscale='log', fit=True,
                yfitstr=r'$v_r$', report=True)
    input = mf[(~mf.t.isnull()) & (mf.t>0) & (mf.peaks>=1)]
    filename=os.path.join(savedir, 'v_t_loglog_peaks1.png')
    scatterplot(input, 't', 'v_pos_abs', 'peaks', filename, xlabel,
                ylabel, xscale='log', yscale='log', fit=True,
                yfitstr=r'$v_r$', report=True)
    filename=os.path.join(savedir, 'v_t_loglog_peaks1_agg.png')
    scatterplot(input, 't', 'v_pos_abs', 'aggregating', filename, xlabel,
                ylabel, xscale='log', yscale='log', fit=True,
                yfitstr=r'$v_r$', report=True)
    input = mf[(~mf.t.isnull()) & (mf.t>0)]
    filename=os.path.join(savedir, 'v_t_agg.png')
    scatterplot(input, 't', 'v_pos_abs', 'aggregating',filename,xlabel,ylabel,
                ylim=(0,4), report=True)

    filename=os.path.join(savedir, 'v_t_agg_loglog.png')
    scatterplot(input, 't', 'v_pos_abs', 'aggregating', filename, xlabel,
                ylabel, xscale='log', yscale='log', fit=True,
                yfitstr=r'$v_r$', report=True)


    # tLOL vs t
    ####################################################################
    ylabel = r'$t_{LOL}$ [µm/s]'
    input = mf[(~mf.t.isnull()) & (mf.t>0) & (mf.peaks>=1)
                &(~mf.tlol_peaks.isnull()) & (mf.tlol_peaks>0)]
    filename=os.path.join(savedir, 'tlol_t.png')
    ylim = (0.01,input.tlol_peaks.max()*1.1)
    xlim = (0.01,input.t.max()*1.1)
    scatterplot(input, 't', 'tlol_peaks', 'peaks',filename,xlabel,
                ylabel, report=True, ylim=ylim, xlim=xlim )
    filename=os.path.join(savedir, 'tlol_t_loglog.png')
    ylim = (0.9*input.tlol_peaks.min(),input.tlol_peaks.max()*1.5)
    xlim = (0.9*input.t.min(),input.t.max()*1.5)
    scatterplot(input, 't', 'tlol_peaks', 'peaks', filename, xlabel,
                ylabel, xscale='log', yscale='log', fit=True,
                yfitstr=r'$t_{LOL}$', report=True, ylim=ylim, xlim=xlim)
    input = mf[(~mf.t.isnull()) & (mf.t>0) & (mf.peaks>=1)]
    filename=os.path.join(savedir, 'tlol_t_agg.png')
    scatterplot(input, 't', 'tlol_peaks', 'aggregating',filename,xlabel,
                ylabel, ylim=ylim, xlim=xlim, report=True)
    filename=os.path.join(savedir, 'tlol_t_agg_loglog.png')
    scatterplot(input, 't', 'tlol_peaks', 'aggregating', filename, xlabel,
                ylabel, xscale='log', yscale='log', fit=True,
                yfitstr=r'$t_{LOL}$', report=True, ylim=ylim, xlim=xlim)


    # v vs tlol
    ####################################################################
    xlabel = r'$t_{LOL}$ [s]'
    ylabel = r'$v_r$ [µm/s]'
    filename=os.path.join(savedir, 'tlol_v.png')
    scatterplot(mf, 'tlol_peaks', 'v_pos_abs', 'peaks', filename, xlabel,
                ylabel, report=True)
    input = mf[(~mf.tlol_peaks.isnull()) & (mf.tlol_peaks>100)  & (~mf.vlol.isin([np.inf]))]
    xlim = (0.9*input.tlol_peaks.min(),input.tlol_peaks.max()*1.5)

    filename=os.path.join(savedir, 'tlol_v_loglog.png')
    scatterplot(input, 'tlol_peaks', 'vlol', 'peaks', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, report=True,
                fit=True, yfitstr=r'$v_r$', xfitstr=r'$t_{LOL}$')
    filename=os.path.join(savedir, 'tlol_v_agg.png')
    scatterplot(mf, 'tlol_peaks', 'v_pos_abs', 'aggregating', filename, xlabel,
                ylabel, report=True)
    filename=os.path.join(savedir, 'tlol_v_agg_loglog.png')
    scatterplot(input, 'tlol_peaks', 'v_pos_abs', 'aggregating', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, report=True,
                fit=True, yfitstr=r'$v_r$', xfitstr=r'$t_{LOL}$')

    filename=os.path.join(savedir, 'tlol_v_agg_peaks_loglog.png')
    input = mf[(~mf.tlol_peaks.isnull()) & (mf.tlol_peaks>0)
                 &(mf.aggregating)]
    scatterplot(input, 'tlol_peaks', 'v_pos_abs', 'peaks', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, report=True,
                fit=True, yfitstr=r'$v_r$', xfitstr=r'$t_{LOL}$')
    filename=os.path.join(savedir, 'tlol_v_nagg_peaks_loglog.png')
    input = mf[(~mf.tlol_peaks.isnull()) & (mf.tlol_peaks>0)
                 &(~mf.aggregating)]
    scatterplot(input, 'tlol_peaks', 'v_pos_abs', 'peaks', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, report=True,
                fit=True, yfitstr=r'$v_r$', xfitstr=r'$t_{LOL}$')

    filename=os.path.join(savedir, 'tlol_vlol1_agg.png')
    scatterplot(mf, 'tlol_peaks', 'v_pos_abslol1', 'aggregating', filename, xlabel,
                ylabel, report=True)
    filename=os.path.join(savedir, 'tlol_vlol1_agg_loglog.png')
    scatterplot(input, 'tlol_peaks', 'v_pos_abslol1', 'aggregating', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, report=True,
                fit=True, yfitstr=r'$v_r$', xfitstr=r'$t_{LOL}$')

    # V vs LOL
    ####################################################################
    xlabel = r'$LOL$'
    ylabel = r'$v_r$ [µm/s]'
    filename=os.path.join(savedir, 'v_lol.png')
    scatterplot(mf, 'xlov_ma_abs_peaks', 'v_pos_abs', 'peaks', filename,
                xlabel, ylabel, report=True)
    filename=os.path.join(savedir, 'v_lol_loglog.png')
    scatterplot(mf, 'xlov_ma_abs_peaks', 'v_pos_abs', 'peaks', filename, xlabel,
                ylabel, yscale='log',
                report=True)

    # Trevesal vs LOL
    ####################################################################
    xlabel = r'$LOL$'
    ylabel = r'$t_{reversal}$ [s]'
    filename=os.path.join(savedir, 'trev_lol.png')
    input = mf[(~mf.trev.isnull()) & (mf.trev>0) & (mf.peaks>=1)
                &(~mf.xlov_ma_abs_peaks.isnull())]
    scatterplot(mf, 'xlov_ma_abs_peaks', 'trev', 'peaks', filename,
                xlabel, ylabel, report=True)
    filename=os.path.join(savedir, 'trev_lol_semilog.png')
    scatterplot(input, 'xlov_ma_abs_peaks', 'trev', 'peaks', filename, xlabel,
                ylabel, yscale='log', report=True)

    # V vs Trevesal
    ####################################################################
    xlabel = r'$t_{reversal}$ [s]'
    ylabel = r'$v_r$ [µm/s]'
    filename=os.path.join(savedir, 'v_trev.png')
    scatterplot(mf, 'trev', 'v_pos_abs', 'peaks', filename, xlabel,
                ylabel, report=True)
    filename=os.path.join(savedir, 'v_trev_loglog.png')
    scatterplot(mf, 'trev', 'v_pos_abs', 'peaks', filename, xlabel,
                ylabel, xscale='log', yscale='log', report=True)

    ylabel = r'$t_{reversal}$ [s]'
    xlabel = r'$t$ [s]'
    filename=os.path.join(savedir, 'trev_t.png')
    scatterplot(mf, 't', 'trev', 'peaks', filename, xlabel,
                ylabel)

    input = mf[(~mf.t.isnull()) & (mf.t>0)]
    xlim = (0.9*input.t.min(),input.t.max()*1.5)
    filename=os.path.join(savedir, 'trev_t_loglog.png')
    scatterplot(input, 't', 'trev', 'peaks', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim)


    # tLOL vs Trevesal
    ####################################################################
    xlabel = r'$t_{reversal}$ [s]'
    ylabel = r'$t_{LOL}$ [s]'
    filename=os.path.join(savedir, 'tlol_trev.png')
    scatterplot(mf, 'trev', 'tlol_peaks', 'peaks', filename, xlabel,
                ylabel)
    input = mf[(~mf.tlol_peaks.isnull()) & (mf.tlol_peaks>0)]
    xlim = (0.9*input.trev.min(),input.trev.max()*1.5)
    ylim = (0.9*input.tlol_peaks.min(),input.tlol_peaks.max()*1.5)

    # V1 vs v0
    ####################################################################

    filename=os.path.join(savedir, 'v0_v1.png')
    scatterplot(mf, 'v_pos_abslol0', 'v_pos_abslol1', 'aggregating', filename, 'v0',
                'v1')
    filename=os.path.join(savedir, 'v0_v1_loglog.png')
    scatterplot(mf, 'v_pos_abslol0', 'v_pos_abslol1', 'aggregating', filename, 'v0',
                'v1', xscale='log', yscale='log')
    """


    # How to increase t?
    ####################################################################

    xlabel = r'$f_{stalling}$'
    ylabel = r'$t$ [s]'
    input = mf[(~mf.t.isnull()) & (mf.t>0) & (mf.fstalling>0)]
    xlim = (0.4*input.fstalling.min(),input.fstalling.max()*1.1)
    ylim = (0.9*input.t.min(),input.t.max()*1.5)
    """
    filename=os.path.join(savedir, 'fstalling.png')
    scatterplot(input, 'fstalling', 't', 'hasPeaks', filename, xlabel,
                ylabel)
    filename=os.path.join(savedir, 'fstalling_agg.png')
    scatterplot(input, 'fstalling', 't', 'aggregating', filename, xlabel,
                ylabel)
    """
    filename=os.path.join(savedir, 'fstalling_loglog.png')
    scatterplot(input, 'fstalling', 't', 'hasPeaks', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, ylim=ylim,
                separate_fit=True, xfitstr='ts', yfitstr='t', report=True)

    filename=os.path.join(savedir, 'fstalling_agg_loglog.png')
    scatterplot(input, 'fstalling', 't', 'aggregating', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, ylim=ylim,
                separate_fit=True, xfitstr='ts', yfitstr='t', report=True)

    filename=os.path.join(savedir, 'fstalling_revb_loglog.png')
    scatterplot(input, 'fstalling', 't', 'revb', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, ylim=ylim,
                separate_fit=True, xfitstr='ts', yfitstr='t', report=True,
                legend_out=True)
    """
    xlabel = r'$t_{stalling}$'
    ylabel = r'$t$ [s]'
    input = mf[(~mf.t.isnull()) & (mf.t>0) & (mf.tstalling>0)]
    xlim = (0.4*input.tstalling.min(),input.tstalling.max()*1.1)
    ylim = (0.9*input.t.min(),input.t.max()*1.5)
    filename=os.path.join(savedir, 'tstalling.png')
    scatterplot(input, 'tstalling', 't', 'hasPeaks', filename, xlabel,
                ylabel)
    filename=os.path.join(savedir, 'tstalling_loglog.png')
    scatterplot(input, 'tstalling', 't', 'hasPeaks', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, ylim=ylim,
                separate_fit=True, xfitstr='ts', yfitstr='t', report=True)
    filename=os.path.join(savedir, 'tstalling_agg.png')
    scatterplot(input, 'tstalling', 't', 'aggregating', filename, xlabel,
                ylabel)
    filename=os.path.join(savedir, 'tstalling_agg_loglog.png')
    scatterplot(input, 'tstalling', 't', 'aggregating', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, ylim=ylim,
                separate_fit=True, xfitstr='ts', yfitstr='t', report=True)
    """

    xlabel = r'$d_{glide}$ [µm]'
    input = mf[(~mf.t.isnull()) & (mf.t>0) & (mf.vtot>0)]
    xlim = (0.9*input.vtot.min(),input.vtot.max()*1.5)
    ylim = (0.9*input.t.min(),input.t.max()*1.5)
    """
    filename=os.path.join(savedir, 'vtot.png')
    scatterplot(input, 'vtot', 't', 'hasPeaks', filename, xlabel,
                ylabel)
    filename=os.path.join(savedir, 'vtot_agg.png')
    scatterplot(input, 'vtot', 't', 'aggregating', filename, xlabel,
                ylabel)
    """
    filename=os.path.join(savedir, 'vtot_loglog.png')
    scatterplot(input, 'vtot', 't', 'hasPeaks', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, ylim=ylim,
                separate_fit=True, xfitstr='d', yfitstr='t', report=True)

    filename=os.path.join(savedir, 'vtot_agg_loglog.png')
    scatterplot(input, 'vtot', 't', 'aggregating', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, ylim=ylim,
                separate_fit=True, xfitstr='d', yfitstr='t', report=True)
    filename=os.path.join(savedir, 'vtot_revb_loglog.png')
    scatterplot(input, 'vtot', 't', 'revb', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, ylim=ylim,
                separate_fit=True, xfitstr='d', yfitstr='t', report=True,
                legend_out=True)

    xlabel = r'$v_r$ [µm/s]'
    input = mf[(~mf.t.isnull()) & (mf.t>0) & (mf.v_pos_abs>0) & (mf.revb!=3)]
    xlim = (0.9*input.v_pos_abs.min(),input.v_pos_abs.max()*1.5)
    ylim = (0.9*input.t.min(),input.t.max()*1.5)
    """
    filename=os.path.join(savedir, 'v.png')
    scatterplot(input, 'v_pos_abs', 't', 'hasPeaks', filename, xlabel,
                ylabel)
    filename=os.path.join(savedir, 'v_agg.png')
    scatterplot(input, 'v_pos_abs', 't', 'aggregating', filename, xlabel,
                ylabel)
    """
    filename=os.path.join(savedir, 'v_loglog.png')
    scatterplot(input, 'v_pos_abs', 't', 'hasPeaks', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, ylim=ylim,
                report=True, m_quantile=True)
    filename=os.path.join(savedir, 'v_agg_loglog.png')
    scatterplot(input, 'v_pos_abs', 't', 'aggregating', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, ylim=ylim,
                report=True, m_quantile=True)
    filename=os.path.join(savedir, 'v_b_loglog.png')
    scatterplot(input, 'v_pos_abs', 't', 'breakup', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, ylim=ylim,
                report=True, m_quantile=True)
    filename=os.path.join(savedir, 'v_brev_loglog.png')
    scatterplot(input, 'v_pos_abs', 't', 'revb', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, ylim=ylim,
                report=True, legend_out=True, m_quantile=True)

    """
    xlabel = r'$t_{LOL}$ [s]'
    input = mf[(~mf.t.isnull()) & (mf.t>0) & (mf.tlol_peaks>0)]
    xlim = (0.9*input.tlol_peaks.min(),input.tlol_peaks.max()*1.5)
    ylim = (0.9*input.t.min(),input.t.dmax()*1.5)
    filename=os.path.join(savedir, 'tlol.png')
    scatterplot(input, 'tlol_peaks', 't', 'peaks', filename, xlabel,
                ylabel)
    filename=os.path.join(savedir, 'tlol_loglog.png')
    scatterplot(input, 'tlol_peaks', 't', 'peaks', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, ylim=ylim,
                report=True, fit=True, xfitstr=r'$t_{LOL}$',
                yfitstr='t')
    filename=os.path.join(savedir, 'tlol_agg.png')
    scatterplot(input, 'tlol_peaks', 't', 'aggregating', filename, xlabel,
                ylabel)
    filename=os.path.join(savedir, 'tlol_agg_loglog.png')
    scatterplot(input, 'tlol_peaks', 't', 'aggregating', filename, xlabel,
                ylabel, xscale='log', yscale='log', xlim=xlim, ylim=ylim,
                report=True, separate_fit=True, xfitstr=r'$t_{LOL}$',
                yfitstr='t')

    """
    xcol = 'v_pos_abs'
    ycol = 'vtot'
    xlabel = r'$v_r$ [µm/s]'
    ylabel = r'$d_{glide}$ [µm]'


    input = mf[(mf[ycol]>0) & (mf[xcol]>=0)]
    ylim = (0.9*input[ycol].min(),input[ycol].max()*1.5)
    xlim = (0.9*input[xcol].min(),input[xcol].max()*1.5)
    filename=os.path.join(savedir, 'dv_agg_loglog.png')
    scatterplot(input, xcol, ycol, 'aggregating', filename, xlabel,
                ylabel, xlim=xlim, ylim=ylim, report=True, yscale='log',
                xscale='log')
    filename=os.path.join(savedir, 'dv_revb_loglog.png')
    scatterplot(input, xcol, ycol, 'revb', filename, xlabel,
                ylabel, xlim=xlim, ylim=ylim, report=True, yscale='log',
                xscale='log')


    mf['nl'] = mf.vtot / (mf.length1 + mf.length2)
    xcol = 'nl'
    ycol = 't'
    ylabel = r'$t$ [s]'
    xlabel = r'$\frac{d_{glide}}{l_1 + l_2}$'
    input = mf[(mf[ycol]>0) & (mf[xcol]>=0)]
    ylim = (0.9*input[ycol].min(),input[ycol].max()*1.5)
    xlim = (0.9*input[xcol].min(),input[xcol].max()*1.5)
    filename=os.path.join(savedir, '{:s}_{:s}_agg_revb.png'.format(ycol, xcol))
    scatterplot(input, xcol, ycol, 'revb', filename, xlabel,
                ylabel, xlim=xlim, ylim=ylim, report=True, yscale='log',
                xscale='log', separate_fit=True, xfitstr='d^*', yfitstr='t',
                legend_out=True)
    filename=os.path.join(savedir, '{:s}_{:s}_agg_loglog.png'.format(ycol, xcol))
    scatterplot(input, xcol, ycol, 'aggregating', filename, xlabel,
                ylabel, xlim=xlim, ylim=ylim, report=True, yscale='log',
                xscale='log', separate_fit=True, xfitstr='d^*', yfitstr='t',
                legend_out=True)

    """
    xcol = 'xlov_ma_abs_peaks'
    ycol = 'tlol_peaks'
    ylabel = r'$v$ [s]'
    xlabel = r'$x_{pos}$'
    input = df[df.tlol_peaks<2000]
    ylim = (1.5*input[ycol].min(),input[ycol].max()*1.5)
    xlim = (1.5*input[xcol].min(),input[xcol].max()*1.5)

    filename=os.path.join(savedir, '{:s}_{:s}_agg_loglog.png'.format(ycol, xcol))
    scatterplot(input, xcol, ycol, 'revb', filename, xlabel,
                ylabel, xlim=xlim, ylim=ylim, report=True)



    mf['frev'] = mf.t / mf.peaks
    xcol = 'peaks'
    ycol = 'frev'
    ylabel = r'$d_{glide}$ [µm]'
    xlabel = r'$n_{reversals}$'
    input = mf[(mf[ycol]>0) & (mf[xcol]>0)]
    ylim = (0.9*input[ycol].min(),input[ycol].max()*1.5)
    xlim = (0.9*input[xcol].min(),input[xcol].max()*1.5)
    filename=os.path.join(savedir, '{:s}_{:s}_agg_loglog.png'.format(ycol, xcol))
    scatterplot(input, xcol, ycol, 'aggregating', filename, xlabel,
                ylabel, xlim=xlim, ylim=ylim, report=True, yscale='log')


    mf['nl'] = mf.vtot / (mf.length1 + mf.length2)
    xcol = 'peaks'
    ycol = 'nl'
    ylabel = r'$t$ [µm]'
    xlabel = r'$n_{length1}$'
    input = mf[(mf[ycol]>0) & (mf[xcol]>0)]
    ylim = (0.9*input[ycol].min(),input[ycol].max()*1.5)
    xlim = (0.9*input[xcol].min(),input[xcol].max()*1.5)
    filename=os.path.join(savedir, '{:s}_{:s}_agg_loglog.png'.format(ycol, xcol))
    scatterplot(input, xcol, ycol, 'aggregating', filename, xlabel,
                ylabel, xlim=xlim, ylim=ylim, report=True, yscale='log',
                xscale='log')
    """
