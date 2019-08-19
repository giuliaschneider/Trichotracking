import argparse
import os
import time
from os.path import abspath, isfile, join, isdir

import numpy as np
from iofiles import find_img
from linking import link, merge
from overlap import (calcOverlap,
                     get_segFunctions,
                     getDist,
                     getIntDark)
from postprocessing import Postprocesser
from segmentation import (calc_chamber_df_ulisetup,
                          getBackground,
                          getChamber,
                          filterParticlesArea,
                          particles_sequence, dilate_border)
from trackkeeper import Aggkeeper, Pairkeeper, Trackkeeper, Pairtrackkeeper


def parse_args(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', help='source folder of images')
    parser.add_argument('--dest', help='destination folder of images')
    parser.add_argument('--plot',
                        help='Flag indicating if picutures are plotted',
                        type=bool,
                        default=False)
    parser.add_argument('--meta', help='file containing meta information')
    args = parser.parse_args(arguments[1:])

    srcDir = args.src
    plot = args.plot

    if args.dest is None:
        args.dest = join(args.src, 'results')
    dest = args.dest
    if not isdir(dest):
        os.mkdir(dest)

    if abspath(args.meta) != args.meta:
        meta = abspath(join(args.src, args.meta))
    else:
        meta = args.meta

    return srcDir, dest, plot, meta


class Processor:

    def __init__(self, arguments):
        (self.srcDir,
         self.dest,
         self.plot,
         self.meta) = parse_args(arguments)

        (self.blur,
         self.border,
         self.dark,
         self.dt,
         self.linkDist,
         self.pxConversion,
         self.timestamp) = self.getMetaInfo()

        (self.background,
         self.dchamber,
         self.trackFile,
         self.pixelFile,
         self.timesFile,
         self.tracksMetaFile,
         self.aggregatesMetaFile,
         self.pairsMetaFile,
         self.pairTrackFile) = self.defineFiles()

        t0 = time.time()
        if not (isfile(self.trackFile)
                and isfile(self.pixelFile)
                and isfile(self.timesFile)
                and isfile(self.tracksMetaFile)
                and isfile(self.aggregatesMetaFile)
                and isfile(self.pairsMetaFile)):
            (self.keeper,
             self.aggkeeper,
             self.pairkeeper,
             self.listTimes) = self.process()

        else:
            self.keeper = Trackkeeper.fromFiles(self.trackFile, self.pixelFile, self.tracksMetaFile)
            self.keeper.update_meta()
            self.aggkeeper = Aggkeeper.fromFile(self.aggregatesMetaFile)
            self.keeper.addMetaTrackType(self.aggkeeper.getDf())
            self.pairkeeper = Pairkeeper.fromFile(self.pairsMetaFile)
            self.listTimes = np.loadtxt(self.timesFile)

        if not isfile(self.pairTrackFile):
            dfPairTracks = self.overlap()
            self.pairTrackKeeper = Pairtrackkeeper(dfPairTracks, self.pairkeeper)
        else:
            self.pairTrackKeeper = Pairtrackkeeper.fromFiles(self.pairTrackFile, self.pairsMetaFile)

        t1 = time.time()
        print("Used time = {} s".format(t1 - t0))

        Postprocesser(self.keeper, self.aggkeeper, self.pairTrackKeeper, self.listTimes, self.dest, self.pxConversion)
        self.save_all()

    def getMetaInfo(self):
        metad = {}
        with open(self.meta) as f:
            for line in f:
                (key, val) = (line.strip().split(': '))
                metad[key] = val
        dark = metad['Darkfield']
        pxConversion = float(metad['pxConversion'])
        linkDist = int(metad['LinkDist'])
        if 'timestamp' in metad.keys():
            timestamp = (metad['timestamp'] == 'True')
            dt = int(metad['dt'])
        else:
            timestamp = True
            dt = None
        if 'blur' in metad.keys():
            blur = True
        else:
            blur = False
        if 'ChamberBorder' in metad.keys():
            border = int(metad['ChamberBorder'])
        else:
            border = 50
        return blur, border, dark, dt, linkDist, pxConversion, timestamp

    def defineFiles(self):
        background = getBackground(self.srcDir)
        chamber = getChamber(self.srcDir, background, calc_chamber_df_ulisetup)
        dchamber = dilate_border(chamber, ksize=self.border)

        trackFile = join(self.dest, 'tracks.csv')
        pixelFile = join(self.dest, 'pixellists.pkl')
        timesFile = join(self.dest, 'times.csv')
        tracksMetaFile = join(self.dest, 'tracks_meta.csv')
        aggregatesMetaFile = join(self.dest, 'aggregates_meta.csv')
        pairsMetaFile = join(self.dest, 'pairs_meta.csv')
        pairsTrackFile = join(self.dest, 'overlap', 'tracks_pair.csv')
        return (
            background, dchamber, trackFile, pixelFile, timesFile, tracksMetaFile, aggregatesMetaFile, pairsMetaFile,
            pairsTrackFile)

    def process(self):
        df, listTimes, dfPixellist = self.segment()
        listTimes = self.get_times(listTimes)
        dfTracks = link(df)
        keeper = Trackkeeper.fromDf(dfTracks, dfPixellist)
        df_merge, df_split = merge(keeper)
        aggkeeper = Aggkeeper.fromScratch(df_merge, df_split, keeper)
        keeper.update_meta()
        keeper.addMetaTrackType(aggkeeper.getDf())
        pairTrackNrs = keeper.getTrackNrPairs()
        pairkeeper = Pairkeeper.fromScratch(aggkeeper.getDf(),
                                            keeper.getDfTracksMeta(),
                                            pairTrackNrs)
        return keeper, aggkeeper, pairkeeper, listTimes

    def segment(self):
        df, listTimes = particles_sequence(self.srcDir,
                                           self.dest,
                                           self.pxConversion,
                                           self.linkDist,
                                           filterParticlesArea,
                                           background=self.background,
                                           plotImages=self.plot,
                                           threshold=30,
                                           roi=self.dchamber,
                                           blur=self.blur,
                                           darkField=self.dark)

        cols = ['pixellist_xcoord', 'pixellist_ycoord', 'contours']
        dfPixellist = df[['index'] + cols]
        df.drop(columns=cols, inplace=True)
        return df, listTimes, dfPixellist

    def get_times(self, listTimes):
        if not self.timestamp:
            listTimes = self.dt * np.arange(len(listTimes))
        if os.path.isfile(self.timesFile):
            listTimes = np.loadtxt(self.timesFile)
        return listTimes

    def overlap(self):
        seg_functions = get_segFunctions(4, 3, (0.7, 3, 5), (0.85, 3, 5),
                                         (0.7, 3, 0.8, 2.8, 6), (0.55, 2.6, 0.65, 2, 6))
        listImgs = find_img(self.srcDir)
        overlap = calcOverlap(listImgs,
                              self.listTimes,
                              self.pairkeeper,
                              self.keeper.getDfTracksComplete(),
                              os.path.join(self.dest, 'overlap'),
                              self.background,
                              getDist,
                              getIntDark,
                              seg_functions,
                              ofs=None,
                              darkphases=None,
                              plotAnimation=True,
                              plotImages=False)
        return overlap.getDf()

    def save_all(self):
        self.keeper.save(self.trackFile, self.pixelFile, self.tracksMetaFile)
        self.keeper.saveAnimation(self.srcDir, self.dest)
        self.aggkeeper.save(self.aggregatesMetaFile)
        self.pairTrackKeeper.save(self.pairTrackFile)
        self.pairkeeper.save(self.pairsMetaFile)
        np.savetxt(self.timesFile, self.listTimes)
