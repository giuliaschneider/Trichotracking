import os.path
import numpy as np
import matplotlib as plt
import seaborn as sns


def plot_all_lol(df, overlapDir, exp= None, hue=None, fnAppendix=None):
    """ Plot lol at peaks, different normalization. """
    if exp is None:
        exp = np.unique(df.exp.values)

    if isinstance(exp, str):
        savedirexp = os.path.join(overlapDir, exp)
        if not os.path.isdir(savedirexp):
            os.mkdir(savedirexp)
        savedir = os.path.join(overlapDir, exp, 'lol')
        expfile = exp
        props = {'hue':'label', 'data':df[(df.exp==exp)&(~df.peaks.isnull())]}
    else:
        savedir = os.path.join(overlapDir, 'lol')
        expfile = ''
        props = {'hue':'exp', 'data':df[(df.exp.isin(exp))&(~df.peaks.isnull())]}

    if hue is not None:
        props['hue'] = hue

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if fnAppendix is not None:
        fnAppendix += '.png'
    else:
        fnAppendix = '.png'

    # LOL (x/l2)
    g = sns.catplot(x='aggregating', y='xlov_ma_abs_peaks', **props)
    g.set_ylabels('LOL reversal')
    g.savefig(os.path.join(savedir, expfile+'_xlol_peaks' + fnAppendix))


    # time LOL (x/vm_means)
    g = sns.catplot(x='aggregating', y='tlol_peaks', **props)
    g.set_ylabels(r'T$_{LOL}$ reversal')
    g.savefig(os.path.join(savedir, expfile+'_tlol_peaks'+fnAppendix))


    g = sns.catplot(x='aggregating', y='tov_peaks', **props)
    g.set_ylabels(r'T$_{overlap}$ reversal')
    g.savefig(os.path.join(savedir, expfile+'_tov_peaks'+fnAppendix))

    g = sns.catplot(x='aggregating', y='lol_normed1', **props)
    g.set_ylabels(r'LOL / l$_{total}$')
    g.savefig(os.path.join(savedir, expfile+'_lol_normed1_peaks.png'))

    g = sns.catplot(x='aggregating', y='lol_normed2', **props)
    g.set_ylabels(r'$\frac{LOL}{l_1 + l_2}$')
    g.savefig(os.path.join(savedir, expfile+'_lol_normed2_peaks'+fnAppendix))
