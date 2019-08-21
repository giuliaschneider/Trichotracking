import numpy as np

from trichotracking.dfmanip import (calcChangeInTime,
                     calcMovingAverages, 
                     calcPeaks,
                     calcSingleFilamentVelocity)
from ._filter_tracks import cleanTracks
from ._import_all import import_all_dfoverlap

def postprocessTracks(dfMeta, expLabels=None):
    """ Importes dataframes in listOfFiles and calculates some variables.

    Returns a dataframe with columns:
    - index     continuous numbering
    - exp       label of experiment
    - label     track label
    - track     track nr
    - dark      True if in darkphase, False otherwise
    - peaks     1 if peak

    - length1   length of filament 1
    - length2   length of filament 2
    - cx1       x-coord of centroid of fil 1
    - cy1       y-coord of centroid of fil 1
    - cx2       x-coord of centroid of fil 2
    - cy2       y-coord of centroid of fil 2
    - dirx1     x direction of unit vector of fil 1
    - diry1     y direction of unit vector of fil 1
    - length_overlap    length of overlap region

    - l1_ma     moving averaged length of filament 1
    - l2_ma     moving averagedlength of filament 2
    - lov_ma    moving averaged length of ovlerap
    - cx1_ma    moving averaged x-coord of centroid of fil 1
    - cy1_ma    moving averaged y-coord of centroid of fil 1
    - cx2_ma    moving averaged x-coord of centroid of fil 2
    - cy2_ma    moving averaged y-coord of centroid of fil 2

    - xlov      absolute length of lack of overlap in long. dir of fil 1
    - ylov      absolute length of lack of overlap in lat. dir of fil 1
    - xlov_norm relative length of lack of overlap in long. dir of fil 1
    - ylov_norm relative length of lack of overlap in lat. dir of fil 1
    - xlov_ma   relative, moving averaged xlov fil 1
    - xlov_ma_abs   absolute relative, moving averaged xlov fil 1
    - xlov_ma_abs_peaks absolute relative ma xlov fil 1 at peaks

    - pos_short relative position of the short fil along long fil
    - pos_ma    moving averaged relative position
    - pos_abs   absolute position of the short fil along long fil
    - pos_abs_ma    moving averaged absolute position

    - v_pos     change in absolute position
    - v1        velocity of fil 1
    - v2        velocity of fil 2    -
    - v_pos_ma  moving averaged change in absolute position
    - v1_ma        moving averaged velocity of fil 1
    - v2_ma        moving averaged velocity of fil 2
    - v_pos_abs absolute v_pos
    - v1_abs    absolute v1
    - v2_abs    absolute v2

    """
    # Read dataframe
    if expLabels is None:
        expLabels = dfMeta.exp.values
    df = import_all_dfoverlap(dfMeta, expLabels)
    #df = df[(~df.length1.isnull()) & (~df.length2.isnull())]

    #df = cleanTracks(df)
    df["xlov_norm"] = df.xlov / df.length2
    df["ylov_norm"] = df.ylov / df.length2
    df['pos_abs'] = df.pos_short * df.length1


    # Calculate moving averages
    columns =    ["length1", "length2","length_overlap", "xlov_norm",
                  "xlov", "pos_short", "pos_abs"]
    ma_columns = ["l1_ma", "l2_ma", "lov_ma", "xlov_ma" ,
                  "xlov_abs_ma", "pos_ma", "pos_abs_ma"]
    df = calcMovingAverages(df, 5, columns, ma_columns)
    columns = ["cx1", "cx2", "cy1", "cy2"]
    ma_columns =  ["cx1_ma", "cx2_ma", "cy1_ma", "cy2_ma"]
    df = calcMovingAverages(df, 11, columns, ma_columns)

    # Calculale peaks
    df = calcPeaks(df, 'pos_abs_ma', p=20)


    # Calculate xlov peaks normed
    df['xlov_ma_abs'] = df.xlov_ma.abs()
    df['xlov_ma_abs_peaks'] = np.nan
    df.loc[((df.peaks==1)),'xlov_ma_abs_peaks'] \
        =    df[((df.peaks==1))].xlov_ma_abs
    # Calculate xlov peaks not normed
    df['xlov_abs_ma_abs_peaks'] = np.nan
    df.loc[((df.peaks==1) & (df.xlov_ma_abs>0.01)),'xlov_abs_ma_abs_peaks'] \
        =    df[((df.peaks==1) & (df.xlov_ma_abs>0.01))].xlov_abs_ma.abs()
    # Calculate overlap of peaks
    df['lov_ma_peaks'] = np.nan
    df.loc[(df.peaks==1),'lov_ma_peaks'] = df[(df.peaks==1)].lov_ma

    # Calculate velocities
    df = calcSingleFilamentVelocity(df)
    columns =      ["pos_abs_ma"]
    diff_columns = ["v_pos"]
    df = calcChangeInTime(df, 'time', columns, diff_columns)
    columns = ["v_pos", "v1", "v2"]
    ma_columns =  ["v_pos_ma", "v1_ma", "v2_ma"]
    df = calcMovingAverages(df, 5, columns, ma_columns)
    df['v_pos_abs'] = df.v_pos_ma.abs()
    df['v1_abs'] = df.v1_ma.abs()
    df['v2_abs'] = df.v2_ma.abs()
    df["v2_v1"] = df.v2_ma - df.v1_ma
    df["v2v1"] = df.v2_ma + df.v1_ma

    df['vlol'] = -np.sign(df.xlov_ma_abs.diff(periods=-1)) * df.v_pos_abs


    # Calculalte acceleration
    columns =      ["v1_ma", "v2_ma"]
    diff_columns = ["a1", "a2"]
    df = calcChangeInTime(df, 'time', columns, diff_columns)

    # Calculte reversal times
    dfg = df.groupby(by=['exp', 'label', 'aggregating'])
    df['vmean'] = dfg['v_pos_abs'].transform('mean')


    df['lol_normed1'] = df.xlov_abs_ma_abs_peaks / (df.l1_ma  + df.l2_ma - df.lov_ma)
    df['lol_normed2'] = df.xlov_abs_ma_abs_peaks / (df.l1_ma  + df.l2_ma)

    # Normalize position
    lf = df.l2_ma / df.l1_ma
    df["pos_norm"] = (df.pos_ma - lf/2) / (1 - lf)

    return df
