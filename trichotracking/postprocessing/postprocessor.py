from os.path import join

from plot import hist_oneQuantity, ts_oneQuantity

from pdb import set_trace

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
        self.generateOverview()
        self.trackAverageProperties()

    def trackAverageProperties(self):
        dfTracks = self.trackkeeper.getDf()
        dfg = dfTracks.groupby('trackNr')
        self.trackkeeper.addColumnMeta(dfg.v_abs.mean().rename('vabs_mean').reset_index())
        npeaks = dfg.peaks.count()
        self.trackkeeper.addColumnMeta(npeaks.rename('n_reversals').reset_index())
        dt = dfg.time.last() - dfg.time.first()
        self.trackkeeper.addColumnMeta((npeaks/dt).rename('f_reversal').reset_index())
        set_trace()


    def generateOverview(self):
        file = join(self.dest, 'exp_overview.txt')
        with open(file, 'w') as f:
            f.write('# single tracks: {:d}'.format(self.singleTrackNrs.size))
            f.write('# pair tracks: {:d}'.format(self.pairTrackNrs.size))


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
        ts_oneQuantity(dfg, 'v_abs', '|v|', filename)
