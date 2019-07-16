

from ._label import set_plotLabel



def average_df(df, dfMeta, col, agg='mean', colNorm=None, aggNorm='mean'):
    """ Groups df, by first averaging over track, then over exp"""

    # Average values over each track
    dfg = df.groupby(['exp', 'label', 'aggregating'], as_index=False).agg(agg)

    # Calculate norming value and add it to dfg
    if colNorm is not None:
        norm = df.groupby(['exp', 'label', 'aggregating']).agg(aggNorm)[colNorm]
        dfg['norm'] =  dfg[col] / norm.values

    # Average values over each track
    dfg_agg = dfg.groupby(['exp', 'aggregating']).agg(['nunique', 'mean', 'std'])
    dfg_agg['plotLabel'] = dfg_agg.apply(
        lambda row: set_plotLabel(row, dfMeta), axis=1)
    return dfg_agg
