

__all__ = ['groupdf', 'reset_col_levels']

def groupdf(df, groupcol='trackNr', aggfunc=['mean', 'std']):
    """ Return grouped dataframe. """
    dfg = df.groupby(groupcol).agg(aggfunc)
    dfg.reset_index(inplace=True)
    dfg.fillna(0, inplace=True)
    reset_col_levels(dfg)
    return dfg


def reset_col_levels(dfg):
    cols = ['_'.join(col) for col in dfg.columns]
    cols = [col[:-1] if col.endswith('_') else col for col in cols]
    cols = [col[1:] if col.startswith('_') else col for col in cols]
    dfg.columns = cols
