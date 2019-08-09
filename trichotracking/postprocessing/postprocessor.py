from os.path import join

from plot import hist_oneQuantity, ts_oneQuantity


class Postprocesser:

    def __init__(self, trackkeeper, aggkeeper, pairkeeper, destDir):
        self.trackkeeper = trackkeeper
        self.aggkeeper = aggkeeper
        self.pairkeeper = pairkeeper
        self.singleTrackNrs = trackkeeper.getTrackNrSingles()
        self.pairTrackNrs = trackkeeper.getTrackNrPairs()
        self.dest = destDir

        self.saveHist_vSingle()
        self.save_ts_vFrame()

    def saveHist_vSingle(self):
        df = self.trackkeeper.getDf()
        df_single = df[df.trackNr.isin(self.singleTrackNrs)]
        cond = (~df_single.v_abs.isnull())
        filename = join(self.dest, 'hist_vsingle.png')
        hist_oneQuantity(df_single, 'v_abs', filename, '|v|', cond1=cond, plotMean=True)
        dfg = df_single.groupby('trackNr')
        vmean = dfg.mean()
        filename = join(self.dest, 'hist_vsingle_grouped.png')
        cond = (~vmean.v_abs.isnull())
        hist_oneQuantity(vmean, 'v_abs', filename, '|v|', cond1=cond, plotMean=True)

    def save_ts_vFrame(self):
        df = self.trackkeeper.getDf()
        df_single = df[df.trackNr.isin(self.singleTrackNrs)]
        dfg = df_single.groupby('frame').mean()
        filename = join(self.dest, 'ts_vsingle.png')

        ts_oneQuantity(dfg, 'v_abs', '¦v¦', filename)
