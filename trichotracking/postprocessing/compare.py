import argparse
import os
import sys
from os.path import join

import numpy as np
import pandas as pd
from plot import plot_vhist, V_UNIT, plot_thist, plot_lolhist, scatterplot
from postprocessing import bars
from trackkeeper import Trackkeeper, Pairtrackkeeper


def parse_args(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', help='source folder')
    args = parser.parse_args(arguments[1:])
    srcDir = args.src
    return srcDir


def defineFiles(dirPath):
    dirPath = join(dirPath, 'results')
    trackFile = join(dirPath, 'tracks.csv')
    pixelFile = join(dirPath, 'pixellists.pkl')
    timesFile = join(dirPath, 'times.csv')
    tracksMetaFile = join(dirPath, 'tracks_meta.csv')
    pairsMetaFile = join(dirPath, 'pairs_meta.csv')
    pairsTrackFile = join(dirPath, 'overlap', 'tracks_pair.csv')
    return trackFile, pixelFile, tracksMetaFile, pairsMetaFile, pairsTrackFile, timesFile


def loadFiles(dirPath):
    trackFile, pixelFile, tracksMetaFile, pairsMetaFile, pairsTrackFile, timesFile = defineFiles(dirPath)
    trackkeeper = Trackkeeper.fromFiles(trackFile, None, tracksMetaFile)
    pairtrackkeeper = Pairtrackkeeper.fromFiles(pairsTrackFile, pairsMetaFile)
    listTimes = np.loadtxt(timesFile)
    return trackkeeper, pairtrackkeeper, listTimes


CDF_PARAMS = {'report': False, 'cdf': False, 'sigTest': True, 'plotMean': True, 'legendMean': True}


class Comparator:

    def __init__(self, arguments):
        self.srcDir = parse_args(arguments)
        self.dest = self.define_destination()
        self.controlDir = join(self.srcDir, 'Control')
        self.menadioneDir = join(self.srcDir, 'Menadione')

        self.cTrackKeeper, self.cPairTrackKeeper, self.cTimes = loadFiles(self.controlDir)
        self.mTrackKeeper, self.mPairTrackKeeper, self.mTimes = loadFiles(self.menadioneDir)

        self.dfMetaTracks, self.dfMetaPairTracks, self.dfPairTracks = self.combine_dataframes()
        self.compare_velocity()
        self.compare_time()
        self.compare_lol()
        self.scatter_plots()
        self.bars()

    def define_destination(self):
        dest = join(self.srcDir, 'compare')
        if not (os.path.isdir(dest)):
            os.mkdir(dest)
        return dest

    def create_label(self, prefix, df):
        return prefix + df.trackNr.astype('str')

    def combine_dataframes(self):
        cMetaTracks = self.cTrackKeeper.meta.getDf()
        cMetaTracks['aggregating'] = False
        cMetaTracks['startTime_t'] = self.cTimes[cMetaTracks.startTime]
        cMetaTracks['endTime_t'] = self.cTimes[cMetaTracks.endTime]
        cMetaTracks['label'] = self.create_label('c', cMetaTracks)

        mMetaTracks = self.mTrackKeeper.meta.getDf()
        mMetaTracks['aggregating'] = True
        mMetaTracks['startTime_t'] = self.mTimes[mMetaTracks.startTime]
        mMetaTracks['endTime_t'] = self.mTimes[mMetaTracks.endTime]
        mMetaTracks['label'] = self.create_label('m', mMetaTracks)

        dfMetaTracks = pd.concat([cMetaTracks, mMetaTracks])
        dfMetaTracks['t'] = dfMetaTracks.endTime_t - dfMetaTracks.startTime_t

        cMetaPairTracks = self.cPairTrackKeeper.meta.getDf()
        cMetaPairTracks['aggregating'] = False
        cMetaPairTracks['label'] = self.create_label('c', cMetaPairTracks)
        mMetaPairTracks = self.mPairTrackKeeper.meta.getDf()
        mMetaPairTracks['aggregating'] = True
        mMetaPairTracks['label'] = self.create_label('m', mMetaPairTracks)
        dfMetaPairTracks = pd.concat([cMetaPairTracks, mMetaPairTracks])

        cPairTracks = self.cPairTrackKeeper.getDf()
        cPairTracks['aggregating'] = False
        cPairTracks['label'] = self.create_label('c', cPairTracks)
        mPairTracks = self.mPairTrackKeeper.getDf()
        mPairTracks['aggregating'] = True
        mPairTracks['label'] = self.create_label('m', mPairTracks)
        dfPairTracks = pd.concat([cPairTracks, mPairTracks])
        return dfMetaTracks, dfMetaPairTracks, dfPairTracks

    def compare_velocity(self):
        # Plot histogram of absolute single filament velocities
        mdfs = self.dfMetaTracks[self.dfMetaTracks.type == 1]
        xlabel = r'$|v_{s}|$' + V_UNIT
        xfitstr = r'|v_s|'
        filename = join(self.dest, "hist_vsingle_cdf.png")
        plot_vhist(mdfs, 'vabs_mean', filename, xlabel, cdf=True, report=False, legend_out=True,
                   sigTest=True, legendMean=True, plotMean=True, xfitstr=xfitstr)
        filename = join(self.dest, "hist_vsingle_nostalling_cdf.png")
        plot_vhist(mdfs, 'vabs_mean_without_stalling', filename, xlabel, cdf=True, report=False, legend_out=True,
                   sigTest=True, legendMean=True, plotMean=True, xfitstr=xfitstr)

        # Plot histogram of relative filament pair velocities
        mdfo = self.dfMetaPairTracks[self.dfMetaPairTracks.couldSegment]
        xlabel = r'$|v_{r}|$' + V_UNIT
        xfitstr = r'|v_{r}|'
        filename = join(self.dest, "hist_voverlap_cdf.png")
        plot_vhist(mdfo, 'relative_v_mean', filename, xlabel, cdf=True, report=False, legend_out=True,
                   sigTest=True, legendMean=True, plotMean=True, xfitstr=xfitstr)

    def compare_time(self):
        xlabel = r'$t$ [s]'
        col = 'time'
        mdfo = self.dfMetaPairTracks[self.dfMetaPairTracks.couldSegment]
        xlim = (1, 4000)
        filename = join(self.dest, "hist_tpair_cdf.png")
        plot_thist(mdfo, col, filename, xlabel, xlim=xlim, **CDF_PARAMS)

        filename = join(self.dest, "hist_tpair_split_cdf.png")
        slabels = mdfo[mdfo.breakup == 2].label.values
        text = 'Separating pairs'
        plot_thist(mdfo, col, filename, xlabel, slabels, xlim=xlim, text=text, **CDF_PARAMS)

        filename = join(self.dest, "hist_tpair_merge_cdf.png")
        slabels = mdfo[mdfo.breakup != 2].label.values
        text = 'Non-eparating pairs'
        plot_thist(mdfo, col, filename, xlabel, slabels, xlim=xlim, text=text, **CDF_PARAMS)

        filename = join(self.dest, "hist_tpair_reversals1_cdf.png")
        slabels = mdfo[mdfo.npeaks > 0].label.values
        text = 'Reversing Pairs'
        plot_thist(mdfo, col, filename, xlabel, slabels, xlim=xlim, text=text, **CDF_PARAMS)

        slabels = mdfo[mdfo.npeaks == 0].label.values
        filename = join(self.dest, "hist_tpair_reversals2_cdf.png")
        text = 'Non-reversing Pairs'
        plot_thist(mdfo, col, filename, xlabel, slabels, text=text, **CDF_PARAMS)

        slabels = mdfo[mdfo.revb == 1].label.values
        filename = join(self.dest, "_hist_tpair_reversals3_cdf.png")
        text = 'Reversing, non-separating pairs'
        plot_thist(mdfo, col, filename, xlabel, slabels, xlim=xlim, text=text, **CDF_PARAMS)

        slabels = mdfo[mdfo.revb == 2].label.values
        filename = join(self.dest, "hist_tpair_reversals4_cdf.png")
        text = 'Reversing, separating pairs'
        plot_thist(mdfo, col, filename, xlabel, slabels, text=text, **CDF_PARAMS)

        slabels = mdfo[mdfo.revb == 4].label.values
        filename = join(self.dest, "hist_tpair_reversals5_cdf.png")
        text = 'Non-reversing, separating pairs'
        plot_thist(mdfo, col, filename, xlabel, slabels, text=text, **CDF_PARAMS)

        slabels = mdfo[mdfo.revb == 3].label.values
        filename = join(self.dest, "hist_tpair_reversals6_cdf.png")
        text = 'Non-reversing, non-separating pairs'
        plot_thist(mdfo, col, filename, xlabel, slabels, text=text, **CDF_PARAMS)

    def compare_lol(self):
        mdfo = self.dfMetaPairTracks[self.dfMetaPairTracks.couldSegment]
        slabels = mdfo.label.values
        df = self.dfPairTracks
        df = df[df.label.isin(slabels)]

        xlabel = r'$|LOL_{rev}|$'
        xfitstr = '|LOL|'

        filename = join(self.dest, "hist_lol_cdf.png")
        plot_lolhist(df, 'lol_reversals_normed', filename, xlabel, xfitstr=xfitstr, **CDF_PARAMS)

        slabels = mdfo[(mdfo.revb == 1)].label.values
        filename = join(self.dest, "hist_lol_cdf1.png")
        text = 'Reversing, non-separating pairs'
        plot_lolhist(df, 'lol_reversals_normed', filename, xlabel, slabels, text=text, xfitstr=xfitstr, **CDF_PARAMS)

        slabels = mdfo[(mdfo.revb == 2)].label.values
        filename = join(self.dest, "hist_lol_cdf2.png")
        text = 'Reversing, separating pairs'
        plot_lolhist(df, 'lol_reversals_normed', filename, xlabel, slabels, text=text, xfitstr=xfitstr, **CDF_PARAMS)

    def scatter_plots(self):
        mf = self.dfMetaPairTracks[self.dfMetaPairTracks.couldSegment]
        xlabel = r'$|LOL_{rev}|$'
        ylabel = r'$|v_{s}|$' + V_UNIT
        xcol = 'lol_reversals_normed_mean'
        ycol = 'relative_v_mean'
        dfin = mf[(~mf[xcol].isnull())
                  & (~mf[ycol].isnull())]

        filename = os.path.join(self.dest, 'scatter_lol_v.png')
        scatterplot(dfin, xcol, ycol, 'aggregating', filename, xlabel, ylabel,
                    ylim=(0, 4), report=False)

    def bars(self):
        bars.bars_of_breakup(mdfo, overlapDir)
        bars.bars_of_agg(mdfo, overlapDir)


if __name__ == '__main__':
    argv = sys.argv
    comparator = Comparator(argv)
