import pandas as pd

from dfmeta import get_metadfo
from ._label import calcLabel

from IPython.core.debugger import set_trace



def import_allmetadfoverlap(dfMeta, expLabels=None):

    df_list = []
    if expLabels is None:
        expLabels = dfMeta.exp.values
    for exp in expLabels:
        metadfoFiles = get_metadfo(dfMeta, exp)
        for i, dffile in enumerate(metadfoFiles):
            df_c = pd.read_csv(dffile)
            df_c['exp'] = exp
            df_c['chamber'] = i+1
            trackNrs = df_c.trackNr.values
            df_c["label"] = calcLabel(trackNrs, i+1, exp)
            df_list.append(df_c)
        print("Experiment " + exp + ": Imported mdfo")

    df = pd.concat(df_list, axis=0, ignore_index=True)
    return df



def get_mdfo(dfMeta, dfo, df_tracks, expLabels=None):
    """ Import the meta dfoverlap file for one chamber."""
    mdfo = import_allmetadfoverlap(dfMeta, expLabels=expLabels)

    # Add mean change in overlap
    #dfg = dfo.groupby('label').median()

    #reversal_labels = dfg[~dfg.peaks.isnull()].label.values
    #nonreversal_labels = dfg[dfg.peaks.isnull()].label.values

    dfg = dfo[dfo.xlov_ma_abs_peaks>0.01].groupby('label')
    lol_peak = dfg['xlov_ma_abs_peaks'].mean()
    mdfo = mdfo.join(lol_peak, on='label')

    dfg = dfo[dfo.v_pos_abs<4].groupby('label')
    meanv = dfg['v_pos_abs'].mean()
    dfg = dfo.groupby('label')
    peaks = dfg['peaks'].count()
    tlol_peak = dfg['tlol_peaks'].mean()
    mdfo = mdfo.join(meanv, on='label')
    mdfo = mdfo.join(peaks, on='label')
    mdfo = mdfo.join(tlol_peak, on='label')
    mdfo['dl'] = mdfo.length1 - mdfo.length2
    # Add time scales and aggregating
    vdf_tracks = df_tracks[['label','aggregating','t','t_norm']].set_index('label')
    mdfo = mdfo.join(vdf_tracks, on='label', how='left')

    return mdfo


def get_mdfs(df_tracks, dflinked):
    mdfs = df_tracks[df_tracks.type==1][['exp','label','t','aggregating']]
    dfg = dflinked.groupby('label')
    meanv = dfg['v_abs'].mean()
    peaks = dfg['peaks'].sum()
    length = dfg['length'].mean()
    mdfs = mdfs.join(meanv, on='label')
    mdfs = mdfs.join(peaks, on='label')
    mdfs = mdfs.join(length, on='label')
    mdfs['hasPeaks'] = mdfs.peaks>=1
    return mdfs
