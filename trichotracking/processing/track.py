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
        df, df_tracks, dfagg, listTimes, filTracks = self.track_and_link()
        self.overlap(df, df_tracks, dfagg, listTimes, filTracks)
        self.pprocess()



    def parse_args(self, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--src', help='source folder of images')
        parser.add_argument('--dest',  help='destination folder of images')
        parser.add_argument('--plot',  help='Flag indicating if picutures are plotted', type=bool, default=False)
        parser.add_argument('--meta',  help='file containing meta information')


        args = parser.parse_args(argv[1:])
        if args.dest is None:
            args.dest = join(args.src, 'results')

        print(args)

        self.srcDir = args.src
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


        self.trackfile = join(self.dest, 'tracks.csv')
        self.timesfile = join(self.dest, 'times.csv')
        self.aggFile = join(self.dest, 'tracks_agg.csv')
        self.dftracksfile = join(self.dest, 'df_tracks.csv')
        self.dfoverlapfile = join(self.dest, 'df_overlap.csv')


    def track_and_link(self):
        if ((not isfile(self.trackfile)) or (not isfile(self.aggFile))):

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

            if not self.timestamp:
                listTimes = self.dt * np.arange(len(listTimes))

                
            if os.path.isfile(self.timesfile):
                listTimes = np.loadtxt(self.timesfile)

            link = linker(df,
                          listTimes,
                          self.srcDir,
                          self.dest)
            df = link.getDf()
            dfagg = link.getDfagg()
            df_tracks = link.getDfTracks()
            filTracks = link.getFilTracks()
        elif (not isfile(self.dftracksfile)):
            df = pd.read_csv(self.trackfile)
            keepTracks = track_keeper(df, listTimes, self.srcDir, self.srcDir)
            keepTracks.update_DfTracks(0)
            dfagg = import_dfagg(self.aggFile)
            filTracks = keepTracks.filterAllTracks(dfagg)
            keepTracks.saveDfTracks()
            df_tracks = keepTracks.getDfTracks()
            keepTracks.saveTrackToText()
        else:
            df = pd.read_csv(self.trackfile)
            df_tracks = pd.read_csv(self.dftracksfile)
            dfagg = import_dfagg(self.aggFile)
            listTimes = np.loadtxt(self.timesfile)
            filTracks = df_tracks[df_tracks.type==2].trackNr.values

        if not 'ews' in df.keys():
            df['ews'] = df.ew2 / df.ew1
        # Groupc
        self.dfg = groupdf(df)

        return df, df_tracks, dfagg, listTimes, filTracks


    def overlap(self, df, df_tracks, dfagg, listTimes, filTracks):
        if (not isfile(self.dfoverlapfile)):
            dfoverlap = create_dfoverlap(filTracks, dfagg, self.dfg, self.dfoverlapfile)
        else:
            dfoverlap = pd.read_csv(self.dfoverlapfile)
        # Calculalte overlap
        filTracks = dfoverlap.trackNr.values
        filLengths = dfoverlap[['length1', 'length2']].values
        seg_functions = get_segFunctions(4, 3, (0.7, 3, 5), (0.85, 3, 5),
                        (0.7, 3, 0.8, 2.8, 6), (0.55, 2.6, 0.65, 2, 6))
        listImgs = find_img(self.srcDir)
        overlap = calcOverlap(listImgs,
                              listTimes,
                              df,
                              filTracks,
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
        df_tracks.loc[df_tracks.trackNr.isin(notFilTracks), 'type']=np.nan
        filTracks = df_tracks[df_tracks.type==2].trackNr.values
        df_tracks.to_csv(self.dftracksfile)



    def pprocess(self):
        df_tracks = pd.read_csv(self.dftracksfile)
        self.single_tracks = df_tracks[df_tracks.type==1].trackNr.values
        self.dflinked = import_dflinked(self.trackfile, self.timesfile, self.px)
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
