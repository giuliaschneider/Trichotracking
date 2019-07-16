import numpy as np
import pandas as pd


def subtract_first(group):
    t0 = group.iloc[0]
    t = pd.to_timedelta(group - t0, unit='s') / np.timedelta64(1, 's')
    return t

def time_to_peak(df):
    df['ind'] = df.xlov_ma_abs > 0.01
    df['grp_index'] = ((df.label != df.label.shift())
                      | (df.ind != df.ind.shift())).cumsum()
    dfg = df.groupby('grp_index').time
    df['tlol_peaks'] = dfg.transform(subtract_first)
    df.loc[(df.xlov_ma_abs_peaks<=0.01) | df.xlov_ma_abs_peaks.isnull(),
          'tlol_peaks'] = np.nan
    df.drop(['ind', 'grp_index'], axis=1, inplace=True)
    return df
