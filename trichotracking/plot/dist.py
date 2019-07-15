import os.path
import numpy as np


def plot_v_dist(df_tracks, overlapDir, text=None):
    df_tracks['frac'] = df_tracks.v_abs / df_tracks.dist_time

    fig, ax = plt.subplots()
    props = {'ax':ax }
    slabels = df_tracks[df_tracks.type==1].label
    d = df_tracks[(df_tracks.label.isin(slabels))
                & (~df_tracks.frac.isin([np.inf, -np.inf, np.nan]))
                & (df_tracks.frac<2)]
    nagg = d[d.aggregating==1].frac.size
    labelagg = r'Aggregating, $n$ = {}'.format(nagg)
    nnagg = d[d.aggregating==0].frac.size
    labelnagg = r'Non aggregating, $n$ = {}'.format(nnagg)
    sns.distplot(d[d.aggregating==1].frac, label=labelagg, **props)
    sns.distplot(d[d.aggregating==0].frac, label=labelnagg, **props)
    ax.set_xlabel(r'$\frac{\bar{|v|}}{(r_t - r_0)/t}$')
    ax.set_ylabel('Density')
    ax.set_xlim([0, 1.1*d.frac.max()])
    if text is not None:
        ax.legend(frameon=False, title=text)
    else:
        ax.legend(frameon=False)

    filename = os.path.join(overlapDir, 'v', '_hist_v_dist.png')
    plt.show()
    fig.savefig(filename)
    plt.close(fig)
