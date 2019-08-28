import os
import threading
import time
from os.path import isfile, join, isdir

import numpy as np

from trichotracking.iofiles import find_img, getTime
from trichotracking.linking import link, merge
from trichotracking.overlap import (calcOverlap,
                                    get_segFunctions,
                                    getDist,
                                    getIntDark)
from trichotracking.postprocessing import Postprocessor
from trichotracking.segmentation import (calc_chamber_df_ulisetup,
                                         getBackground,
                                         getChamber,
                                         filterParticlesArea,
                                         particles_sequence, dilate_border)
from trichotracking.trackkeeper import Aggkeeper, Pairkeeper, Trackkeeper, Pairtrackkeeper


class Processor:

    def __init__(self, srcDir, px, dest, plot, dark, blur, threshold, dLink, dMerge, dMergeBox, kChamber, dt):
        self.srcDir = srcDir
        self.px = px
        self.dest = dest
        self.plot = plot
        self.dark = dark
        self.blur = blur
        self.threshold = threshold
        self.dLink = dLink
        self.dMerge = dMerge
        self.dMergeBox = dMergeBox
        self.kChamber = kChamber
        self.dt = dt

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
                and isfile(self.aggregatesMetaFile)):
            (self.keeper,
             self.aggkeeper,
             self.listTimes) = self.process()
            self.save_all()

        else:
            self.keeper = Trackkeeper.fromFiles(self.trackFile, self.pixelFile, self.tracksMetaFile)
            self.keeper.update_meta()
            self.aggkeeper = Aggkeeper.fromFile(self.aggregatesMetaFile)
            self.keeper.addMetaTrackType(self.aggkeeper.getDf())
            self.listTimes = np.loadtxt(self.timesFile)

        if not (isfile(self.pairsMetaFile)):
            pairTrackNrs = self.keeper.getTrackNrPairs()
            self.pairkeeper = Pairkeeper.fromScratch(self.aggkeeper.getDf(),
                                                     self.keeper.getDfTracksMeta(),
                                                     pairTrackNrs)
        else:
            self.pairkeeper = Pairkeeper.fromFile(self.pairsMetaFile)

        if not isfile(self.pairTrackFile):
            dfPairTracks = self.overlap()
            self.pairTrackKeeper = Pairtrackkeeper(dfPairTracks, self.pairkeeper)
            self.save_pairs()
        else:
            self.pairTrackKeeper = Pairtrackkeeper.fromFiles(self.pairTrackFile, self.pairsMetaFile)

        t1 = time.time()
        print("Used time = {} s".format(t1 - t0))

        Postprocessor(self.keeper, self.aggkeeper, self.pairTrackKeeper, self.listTimes, self.dest, self.px)

    def defineFiles(self):
        background = getBackground(self.srcDir)
        chamber = getChamber(self.srcDir, background, calc_chamber_df_ulisetup)
        dchamber = dilate_border(chamber, ksize=self.kChamber)

        if not isdir(self.dest):
            os.mkdir(self.dest)

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
        df, dfPixellist = self.segment()
        listTimes = self.get_times()
        dfTracks = link(df, maxLinkDist=self.dLink)
        keeper = Trackkeeper.fromDf(dfTracks, dfPixellist, self.pixelFile)

        df_merge, df_split = merge(keeper,
                                   maxMergeDistBox=self.dMergeBox,
                                   maxMergeDist=self.dMerge)
        aggkeeper = Aggkeeper.fromScratch(df_merge, df_split, keeper)
        keeper.update_meta()
        keeper.addMetaTrackType(aggkeeper.getDf())
        return keeper, aggkeeper, listTimes

    def segment(self):
        df = particles_sequence(self.srcDir,
                                self.dest,
                                self.px,
                                self.dLink,
                                filterParticlesArea,
                                background=self.background,
                                plotImages=self.plot,
                                threshold=self.threshold,
                                roi=self.dchamber,
                                blur=self.blur,
                                darkField=self.dark)

        cols = ['contours']
        dfPixellist = df[['index', ] + cols]
        df.drop(columns=cols, inplace=True)
        return df, dfPixellist

    def get_times(self):
        if os.path.isfile(self.timesFile):
            listTimes = np.loadtxt(self.timesFile)
        elif self.dt is not None:
            listImgs = find_img(self.srcDir, mustKw='.jpg')
            listTimes = self.dt * np.arange(len(listImgs))
        else:
            listImgs = find_img(self.srcDir, mustKw='.jpg')
            listTimes = [getTime(img) for img in listImgs]
            listTimes = np.array(listTimes)

        return listTimes

    def overlap(self):
        seg_functions = get_segFunctions(4, 3, (0.7, 3, 5), (0.85, 3, 5),
                                         (0.7, 3, 0.8, 2.8, 6), (0.55, 2.6, 0.65, 2, 6))
        listImgs = find_img(self.srcDir)
        overlap = calcOverlap(listImgs,
                              self.listTimes,
                              self.pairkeeper,
                              self.keeper.getDfTracksComplete(self.keeper.getTrackNrPairs()),
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
        self.aggkeeper.save(self.aggregatesMetaFile)
        np.savetxt(self.timesFile, self.listTimes)
        animation_thread = threading.Thread(target=self.keeper.saveAnimation, args=(self.srcDir, self.dest))
        animation_thread.start()

    def save_pairs(self):
        self.pairTrackKeeper.save(self.pairTrackFile)
        self.pairkeeper.save(self.pairsMetaFile)
