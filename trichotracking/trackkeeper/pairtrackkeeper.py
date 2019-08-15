import numpy as np
import pandas as pd


from dfmanip import convertPxToMeter, calcMovingAverages, calcSingleFilamentVelocity, calcChangeInTime, calcPeaks
from .pairkeeper import Pairkeeper


class Pairtrackkeeper:
    """ Class storing and providing information about filament pair tracks.

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

    def __init__(self, df, meta):
        self.df = df
        self.meta = meta

        if not("xlov_norm" in self.df.keys()):
            self.df["xlov_norm"] = self.df.xlov / self.df.length2


    @classmethod
    def fromDf(cls, df, meta):
        return cls(df, meta)

    @classmethod
    def fromFiles(cls, tracksPairFile, metaPairFile):
        df = pd.read_csv(tracksPairFile)
        meta = Pairkeeper.fromFile(metaPairFile)
        return cls(df, meta)

    def addColumnMeta(self, df_new):
        self.meta.addColumn(df_new)

    def calcLengthVelocity(self, pxConversion):
        pxCols = ["length1", "length2", "length_overlap", "cx1", "cy1", "cx2", "cy2"]
        umCols = ["l1_um", "l2_um", "lo_um", "cx1_um", "cy1_um", "cx2_um", "cy2_um"]
        if not all(x in self.df.keys() for x in umCols):
            self.df = convertPxToMeter(self.df, pxCols, umCols, pxConversion)

        if not("pos_abs" in self.df.keys()):
            self.df['pos_abs'] = self.df.pos_short * self.df.l1_um

        columns = ["cx1_um", "cy1_um", "cx2_um", "cy2_um", "pos_abs", "xlov_norm", "xlov"]
        ma_columns = ["cx1_ma", "cx2_ma", "cy1_ma", "cy2_ma", "pos_abs_ma", "xlov_ma", "xlov_abs_ma"]
        if not all(x in self.df.keys() for x in ma_columns):
            self.df = calcMovingAverages(self.df, 11, columns, ma_columns)
            self.df['xlov_ma_abs'] = self.df.xlov_ma.abs()

        if not all(x in self.df.keys() for x in ['v1', 'v2']):
            self.df = calcSingleFilamentVelocity(self.df)

        columns = ["pos_abs_ma"]
        diff_columns = ["v_pos"]
        if not all(x in self.df.keys() for x in diff_columns):
            self.df = calcChangeInTime(self.df, 'time', columns, diff_columns)

        columns = ["v_pos", "v1", "v2"]
        ma_columns = ["v_pos_ma", "v1_ma", "v2_ma"]
        if not all(x in self.df.keys() for x in ma_columns):
            self.df = calcMovingAverages(self.df, 5, columns, ma_columns)
        if not all(x in self.df.keys() for x in ['v_pos_abs']):
            self.df['v_pos_abs'] = self.df.v_pos_ma.abs()

    def saveValueAtReversal(self, col, new_col, cond=None):
        self.df[new_col] = np.nan
        if cond is None:
            cond = (self.df.peaks == 1)
        else:
            cond = (cond & (self.df.peaks == 1))
        self.df.loc[cond, new_col] = self.df[cond][col]

    def calcReversals(self, pxConversion):
        if not("peaks" in self.df.keys()):
            self.df = calcPeaks(self.df, 'pos_abs_ma', p=20)

        if not all(x in self.df.keys() for x in ['xlov_ma_abs', 'xlov_abs_ma']):
            self.calcLengthVelocity(pxConversion)
        self.saveValueAtReversal('xlov_ma_abs', 'lol_reversals_normed', cond=(self.df.xlov_ma_abs > 0.01))
        self.saveValueAtReversal('xlov_abs_ma', 'lol_reversals', cond=(self.df.xlov_ma_abs > 0.01))
        self.df['lol_reversals_normed'] = self.df['lol_reversals_normed'].abs()
        self.df['vlol'] = -np.sign(self.df.xlov_ma_abs.diff(periods=-1)) * self.df.v_pos_abs

    def setTime(self, times):
        self.df['time'] = times[self.df.frame]

    def setLabel(self):
        self.df['label'] = self.df.trackNr

    def getDf(self):
        return self.df

    def save(self, file):
        self.df.to_csv(file)

    def getTrackNrPairs(self):
        return self.meta.getSuccessfulTrackNr()

    def add_revb(self):
        self.meta.add_revb()
