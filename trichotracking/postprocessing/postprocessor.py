from os.path import join

from plot import hist_oneQuantity, ts_oneQuantity


class Postprocesser:

    def __init__(self, trackkeeper, aggkeeper, pairtrackskeeper, listTimes, destDir, pxConversion):
        self.trackkeeper = trackkeeper
        self.aggkeeper = aggkeeper
        self.pairtrackskeeper = pairtrackskeeper
        self.listTimes = listTimes
        self.dest = destDir
        self.pxConversion = pxConversion

        self.singleTrackNrs = trackkeeper.getTrackNrSingles()
        self.pairTrackNrs = pairtrackskeeper.getTrackNrPairs()

        self.calculate_trackkeeper()
        self.calculate_pairtrackkeeper()
        self.trackAverageProperties()

        self.saveHist_vSingle()
        self.save_ts_vFrame()
        self.generateOverview()

    def calculate_trackkeeper(self):
        self.trackkeeper.setTime(self.listTimes)
        self.trackkeeper.setLabel()
        self.trackkeeper.calcLengthVelocity(self.pxConversion)
        self.trackkeeper.calcReversals()

    def calculate_pairtrackkeeper(self):
        self.pairtrackskeeper.setTime(self.listTimes)
        self.pairtrackskeeper.setLabel()
        self.pairtrackskeeper.calcLengthVelocity(self.pxConversion)
        self.pairtrackskeeper.calcReversals(self.pxConversion)

    def trackAverageProperties(self):
        dfTracks = self.trackkeeper.getDf()
        dfg = dfTracks.groupby('trackNr')
        self.trackkeeper.addColumnMeta(dfg.v_abs.mean().rename('vabs_mean').reset_index())
        npeaks = dfg.peaks.count()
        self.trackkeeper.addColumnMeta(npeaks.rename('n_reversals').reset_index())
        dt = dfg.time.last() - dfg.time.first()
        self.trackkeeper.addColumnMeta((npeaks / dt).rename('f_reversal').reset_index())

        dfg_nostalling = dfTracks[dfTracks.v_abs > 0.1].groupby('trackNr')
        self.trackkeeper.addColumnMeta(dfg_nostalling.v_abs.mean().rename('vabs_mean_without_stalling').reset_index())
        dfg_stalling = dfTracks[dfTracks.v_abs < 0.1].groupby('trackNr')
        fstalling = dfg_stalling.time.count() / dfg.time.count()
        self.trackkeeper.addColumnMeta(fstalling.rename('fstalling').reset_index())

        dfPairTracks = self.pairtrackskeeper.getDf()
        dfgpairs = dfPairTracks.groupby('trackNr')
        t = (dfgpairs.time.last() - dfgpairs.time.first())
        self.pairtrackskeeper.addColumnMeta(t.reset_index())
        self.pairtrackskeeper.addColumnMeta(dfgpairs.v1_ma.mean().rename('v1_mean').reset_index())
        self.pairtrackskeeper.addColumnMeta(dfgpairs.v2_ma.mean().rename('v2_mean').reset_index())
        self.pairtrackskeeper.addColumnMeta(dfgpairs.v_pos_abs.mean().rename('relative_v_mean').reset_index())
        npeaks = dfgpairs.peaks.count()
        dt = dfgpairs.time.last() - dfgpairs.time.first()
        self.pairtrackskeeper.addColumnMeta(npeaks.rename('npeaks').reset_index())
        self.pairtrackskeeper.add_revb()
        self.pairtrackskeeper.addColumnMeta((npeaks / dt).rename('f_reversal').reset_index())
        self.pairtrackskeeper.addColumnMeta(dfgpairs.lol_reversals.mean().rename('lol_reversals_mean').reset_index())
        self.pairtrackskeeper.addColumnMeta(
            dfgpairs.lol_reversals_normed.mean().rename('lol_reversals_normed_mean').reset_index())
        totallength = dfgpairs.l1_um.mean() + dfgpairs.l2_um.mean()
        self.pairtrackskeeper.addColumnMeta(totallength.rename('total_length').reset_index())

        dfg_nostalling = dfPairTracks[dfPairTracks.v_pos_abs > 0.1].groupby('trackNr')
        self.pairtrackskeeper.addColumnMeta(
            dfg_nostalling.v_pos_abs.mean().rename('rel_v_mean_without_stalling').reset_index())
        dfg_stalling = dfPairTracks[dfPairTracks.v_pos_abs < 0.1].groupby('trackNr')
        fstalling = dfg_stalling.time.count() / dfg.time.count()
        self.pairtrackskeeper.addColumnMeta(fstalling.rename('fstalling').reset_index())

    def generateOverview(self):
        file = join(self.dest, 'exp_overview.txt')
        df = self.pairtrackskeeper.meta.getDf()
        df = df[df.couldSegment].copy()
        df['hasPeaks'] = df.npeaks > 0
        df['separates'] = df.breakup == 2

        nPairs = self.pairTrackNrs.size
        with open(file, 'w') as f:
            f.write('# single tracks: {:d} \n'.format(self.singleTrackNrs.size))
            f.write('# pair tracks: {:d}\n'.format(nPairs))
            ns = df.groupby('hasPeaks').count().trackNr.reset_index()
            f.write('  {:d} reversing\n'.format(100 * ns[ns.hasPeaks].trackNr / nPairs))
            f.write('  {:d} not reversing\n'.format(100 * ns[~ns.hasPeaks].trackNr / nPairs))
            ns = df.groupby('separates').count().trackNr.reset_index()
            f.write('  {:d} separating \n'.format(100 * ns[ns.separates].trackNr / nPairs))
            f.write('  {:d} not separating\n'.format(100 * ns[~ns.separates].trackNr / nPairs))

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
