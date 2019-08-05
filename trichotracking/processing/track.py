import sys
import os
from os.path import abspath, isfile, join

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as pltgetBackground


from dfmanip import groupdf
from iofiles import find_img
from iofiles._import_dfs import import_dfagg
from linking import linker, track_keeper
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


from IPython.core.debugger import set_trace



class ProcessExperiment():


    def __init__(self, argv):
        self.parse_args(argv)
        self.getMetaInfo()
        self.getFiles()
        self.overlap()
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
        self.plot = args.plot

        if not os.path.isdir(self.dest):
            os.mkdir(self.dest)

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

        if not isfile(self.trackFile):
            df, listTimes, dfPixellist = self.segment()
            listTimes = self.get_times(listTimes)
            dfTracks, dfTracksMeta, dfAggMeta = self.link(df, listTimes)
        else:
            dfTracks = pd.read_csv(self.trackFile)
            dfPixellist = pd.read_pickle(self.pixelFile)
            listTimes = np.loadtxt(self.timesFile)
            dfTracksMeta = pd.read_csv(self.tracksMetaFile)
            dfAggMeta = import_dfagg(self.aggregatesMetaFile)

        pairTrackNrs = dfTracksMeta[dfTracksMeta.type==2].trackNr.values

        if (not isfile(self.pairsMetaFile)):
            dfPairMeta = create_dfoverlap(pairTrackNrs, 
                                           dfAggMeta, 
                                           groupdf(dfTracks), 
                                           self.pairsMetaFile)
        else:
            dfPairMeta = pd.read_csv(self.pairsMetaFile)

        self.dfTracks = dfTracks
        self.dfPixellist = dfPixellist
        self.listTimes = listTimes
        self.dfTracksMeta = dfTracksMeta
        self.dfAggMeta = dfAggMeta
        self.dfPairMeta = dfPairMeta
        self.pairTrackNrs = pairTrackNrs

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
        dfPixellist.to_pickle(self.pixelFile)
        df.drop(columns=cols, inplace=True)
        return df, listTimes, dfPixellist

    def get_times(self, listTimes):
        if not self.timestamp:
                listTimes = self.dt * np.arange(len(listTimes))
        if os.path.isfile(self.timesFile):
            listTimes = np.loadtxt(self.timesFile)
        return listTimes

    def link(self, df, listTimes):
        link = linker(df,
                      listTimes,
                      self.srcDir,
                      self.dest)
        dfTracks = link.getDf()
        dfTracksMeta = link.getDfTracks()
        dfAggMeta = link.getDfagg()
        return dfTracks, dfTracksMeta, dfAggMeta

    def overlap(self):
        filLengths = self.dfPairMeta[['length1', 'length2']].values
        seg_functions = get_segFunctions(4, 3, (0.7, 3, 5), (0.85, 3, 5),
                        (0.7, 3, 0.8, 2.8, 6), (0.55, 2.6, 0.65, 2, 6))
        listImgs = find_img(self.srcDir)
        set_trace()
        overlap = calcOverlap(listImgs,
                              self.listTimes,
                              self.dfTracks.merge(self.dfPixellist, left_on='index', right_on='index'),
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
        self.dfTracksMeta.loc[self.dfTracksMeta.trackNr.isin(notFilTracks), 'type']=np.nan
        self.dfPairMeta = self.dfTracksMeta[self.dfTracksMeta.type==2].trackNr.values
        self.dfTracksMeta.to_csv(self.tracksMetaFile)



    def pprocess(self):
        df_tracks = pd.read_csv(self.tracksMetaFile)
        self.single_tracks = df_tracks[df_tracks.type==1].trackNr.values
        self.dflinked = import_dflinked(self.trackFile, self.timesFile, self.px)
        df_single = self.dflinked[self.dflinked.trackNr.isin(self.single_tracks)]
        cond = (~df_single.v_abs.isnull())
        filename = join(self.srcDir, 'hist_vsingle.png')
        hist_oneQuantity(df_single, 'v_abs', filename, '|v|', cond1=cond, plotMean=True)
        dfg = df_single.groupby('trackNr')

        vmean = dfg.mean()
        filename = join(self.srcDir, 'hist_vsingle_grouped.png')
        cond = (~vmean.v_abs.isnull())

        hist_oneQuantity(vmean, 'v_abs', filename, '|v|', cond1=cond, plotMean=True)


        
    

if __name__ == '__main__':
    argv = sys.argv
    exp = ProcessExperiment(argv)

    dflinked = exp.dflinked
