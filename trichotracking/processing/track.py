import sys
import os
from os.path import abspath, isfile, join
import time


import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from dfmanip import groupdf
from iofiles import find_img
from linking import link, link_part, merge
from overlap import (calcOverlap,
                     create_dfoverlap,
                     get_filLengths,
                     get_segFunctions,
                     getDist,
                     getIntLightBlurred,
                     getIntDark)
from plot._hist import hist_oneQuantity
from postprocess.trackfile import import_dflinked
from segmentation import (calc_chamber_df_ulisetup,
                          dilate_border,
                          getBackground,
                          getChamber,
                          filterParticlesArea,
                          particles_sequence)

from trackkeeper import Aggkeeper, Trackkeeper


from IPython.core.debugger import set_trace



class ProcessExperiment():


    def __init__(self, argv):
        self.parse_args(argv)
        self.getMetaInfo()
        self.getFiles()
        t0 = time.time()
        self.process()
        self.overlap()
        t1 = time.time()
        print("Used time = {} s".format(t1-t0))
        self.pprocess()


    def parse_args(self, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--src', help='source folder of images')
        parser.add_argument('--dest',  help='destination folder of images')
        parser.add_argument('--plot',  help='Flag indicating if picutures are plotted', type=bool, default=False)
        parser.add_argument('--meta',  help='file containing meta information')
        args = parser.parse_args(argv[1:])
        
        self.srcDir  = args.src
        
        if args.dest is None:
            args.dest = join(args.src, 'results')
        self.dest = args.dest
        if not os.path.isdir(self.dest):
            os.mkdir(self.dest)

        self.plot = args.plot

        if abspath(args.meta) != args.meta:
            self.meta = abspath(join(args.src, args.meta))
        else:
            self.meta = args.meta

    
    def getMetaInfo(self):
        metad = {}
        with open(self.meta) as f:
            for line in f:
                (key, val) = (line.strip().split(': '))
                metad[(key)] = val
        self.dark = metad['Darkfield']
        self.blur = True
        self.px = float(metad['pxConversion'])
        self.linkDist = int(metad['LinkDist'])
        if 'timestamp' in metad.keys():
            self.timestamp = (metad['timestamp'] == 'True')
            self.dt = int(metad['dt'])
        else:
            self.timestamp = True

        if 'blur' in metad.keys():
            self.blur = True
        else:
            self.blur = False

        if 'ChamberBorder' in metad.keys():
            self.border = int(metad['ChamberBorder'])
        else:
            self.border = 50

        self.linkDist = int(metad['LinkDist'])


    def getFiles(self):
        self.background = getBackground(self.srcDir)
        self.chamber = getChamber(self.srcDir, self.background, calc_chamber_df_ulisetup)
        self.dchamber = dilate_border(self.chamber, ksize=self.border)

        self.trackFile = join(self.dest, 'tracks.csv')
        self.pixelFile = join(self.dest, 'pixellists.pkl')
        self.timesFile = join(self.dest, 'times.csv')
        self.tracksMetaFile = join(self.dest, 'tracks_meta.csv')
        self.aggregatesMetaFile = join(self.dest, 'aggregates_meta.csv')
        self.pairsMetaFile = join(self.dest, 'pairs_meta.csv')

    

    def process(self):
        if not isfile(self.trackFile):
            df, listTimes, dfPixellist = self.segment()
            listTimes = self.get_times(listTimes)
            dfTracks = link(df)
            keeper = Trackkeeper.fromDf(dfTracks, dfPixellist)
            df_merge, df_split = merge(keeper)
            aggkeeper = Aggkeeper.fromScratch(df_merge, df_split, keeper)
            keeper.addMetaTrackType(aggkeeper.df)

        else:
            keeper = Trackkeeper.fromFiles(self.trackFile,  
                                           self.pixelFile,
                                           self.tracksMetaFile)
            listTimes = np.loadtxt(self.timesFile)
            aggkeeper = Aggkeeper.fromFile(self.aggregatesMetaFile)

        pairTrackNrs = keeper.getTrackNrPairs()

        if (not isfile(self.pairsMetaFile)):
            dfPairMeta = create_dfoverlap(pairTrackNrs, 
                                           aggkeeper.df, 
                                           groupdf(keeper.df), 
                                           self.pairsMetaFile)
        else:
            dfPairMeta = pd.read_csv(self.pairsMetaFile)

        self.keeper = keeper
        self.aggkeeper = aggkeeper
        self.listTimes = listTimes
        self.dfPairMeta = dfPairMeta
        self.pairTrackNrs = pairTrackNrs

        keeper.save(self.trackFile, self.pixelFile, self.tracksMetaFile)
        keeper.saveAnimation(self.srcDir, self.dest)
        aggkeeper.save(self.aggregatesMetaFile)

        self.dfPairMeta.to_csv()

    def segment(self):
        df, listTimes = particles_sequence( self.srcDir,
                                                self.dest,
                                                self.px,
                                                self.linkDist,
                                                filterParticlesArea,
                                                background=self.background,
                                                plotImages=self.plot,
                                                threshold=23,
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
        filLengths = self.dfPairMeta[['length1', 'length2']].values
        seg_functions = get_segFunctions(4, 3, (0.7, 3, 5), (0.85, 3, 5),
                        (0.7, 3, 0.8, 2.8, 6), (0.55, 2.6, 0.65, 2, 6))
        listImgs = find_img(self.srcDir)
        overlap = calcOverlap(listImgs,
                              self.listTimes,
                              self.keeper.getDfTracksComplete(),
                              self.pairTrackNrs,
                              self.dest,
                              self.background,
                              getDist,
                              getIntDark,
                              seg_functions,
                              ofs=None,
                              darkphases=None,
                              plotAnimation=True,
                              plotImages=False,
                              filLengths=filLengths)

        notFilTracks = overlap.getUnsuccessfulTracks()
       # self.dfTracksMeta.loc[self.dfTracksMeta.trackNr.isin(notFilTracks), 'type']=np.nan
       # self.dfPairMeta = self.dfTracksMeta[self.dfTracksMeta.type==2].trackNr.values
       # self.dfTracksMeta.to_csv(self.tracksMetaFile)



    def pprocess(self):
        df_tracks = pd.read_csv(self.tracksMetaFile)
        self.single_tracks = df_tracks[df_tracks.type==1].trackNr.values
        self.dflinked = import_dflinked(self.trackFile, self.timesFile, self.px)
        df_single = self.dflinked[self.dflinked.trackNr.isin(self.single_tracks)]
        cond = (~df_single.v_abs.isnull())
        filename = join(self.dest, 'hist_vsingle.png')
        hist_oneQuantity(df_single, 'v_abs', filename, '|v|', cond1=cond, plotMean=True)
        dfg = df_single.groupby('trackNr')

        vmean = dfg.mean()
        filename = join(self.dest, 'hist_vsingle_grouped.png')
        cond = (~vmean.v_abs.isnull())

        hist_oneQuantity(vmean, 'v_abs', filename, '|v|', cond1=cond, plotMean=True)


        
    

if __name__ == '__main__':
    argv = sys.argv
    exp = ProcessExperiment(argv)

    dflinked = exp.dflinked
