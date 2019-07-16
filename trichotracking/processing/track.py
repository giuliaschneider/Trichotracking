import sys
import os
from os.path import isfile, join

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from iofiles import find_img
from linking import linker, track_keeper
from overlap import (calcOverlap,
                     create_dfoverlap,
                     get_filLengths,
                     get_segFunctions,
                     getDist,
                     getIntLightBlurred,
                     getIntDark)
from overlap_analysis import postprocessTracks, import_alldflinked
from overlap_analysis.plot import plotTracks
from plot import plot_vts
from segmentation import (calc_chamber_df_ulisetup,
                          dilate_border,
                          getBackground,
                          getChamber)

from tracking import (findTrichomesArea,
                      segementTrichosBlurred,
                      segementTrichosDarkField,
                      tracker)

sys.path.append("/home/giu/Documents/10Masterarbeit/code/Tricho/scripts")

from IPython.core.debugger import set_trace



class ProcessExperiment():


    def __init__(self, argv):
        self.parse_args(argv)
        self.getFiles()
        df, listImgs, listTimes = self.track()
        df, df_tracks, dfagg, listTimes, filTracks = self.link(df, listImgs, listTimes)
        self.overlap(df, df_tracks, dfagg, listTimes, filTracks, listImgs)



    def parse_args(self, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--src', help='source folder of images')
        parser.add_argument('--dest',  help='destination folder of images')
        parser.add_argument('--px',  help='pixel resolution [µm/px]', type=int, default=1)
        parser.add_argument('--R',  help='maximal link radius between subsequent frames', type=int, default=15)
        parser.add_argument('--plot',  help='Flag indicating if picutures are plotted', type=bool, default=False)

        args = parser.parse_args(argv[1:])
        if args.dest is None:
            args.dest = args.src

        print(args)

        self.srcDir = args.src
        self.dest = args.dest
        self.px = args.px
        self.linkDist = args.R
        self.plot = args.plot


    def getFiles(self):
        _, basename = os.path.split(os.path.normpath(self.srcDir))
        self.background = getBackground(self.srcDir)
        self.chamber = getChamber(self.srcDir, self.background, calc_chamber_df_ulisetup)
        self.dchamber = dilate_border(self.chamber, ksize=200)
        self.trackfile = join(self.srcDir, 'tracks.txt')
        self.timesfile = join(self.srcDir, 'times.txt')
        self.linkedfile = join(self.srcDir, 'tracks_linked.txt')
        self.aggFile = join(self.srcDir, 'tracks_agg.txt')
        self.dftracksfile = join(self.srcDir, 'df_tracks.txt')
        self.dfoverlapfile = join(self.srcDir, 'df_oisfile(verlap.txt')


    def track(self):
        if (not isfile(self.trackfile)) or (not isfile(self.timesfile)):
            track = tracker(self.srcDir,
                            self.dest,
                            self.px,
                            self.linkDist,
                            findTrichomesArea,
                            segementTrichosDarkField,
                            background=self.background,
                            plotImages=self.plot,
                            plotContours=self.plot,
                            plotAnimation=False,
                            threshold=45,
                            roi=self.dchamber)
            df = track.getParticleTracks()
            listTimes = track.getTimes()
            listImgs = track.getListOfImgs()
        else:
            df = pd.read_csv(self.trackfile)
            listTimes = np.loadtxt(self.timesfile)
            listImgs = find_img(self.srcDir)
        return df, listImgs, listTimes



    def link(self, df, listImgs, listTimes):
        if ((not isfile(self.linkedfile)) or (not isfile(self.aggFile))):
            link = linker(df,
                          listTimes,
                          self.srcDir,
                          self.srcDir)
            df = link.getDf()
            dfagg = link.getDfagg()
            df_tracks = link.getDfTracks()
            filTracks = link.getFilTracks()
        elif (not isfile(self.dftracksfile)):
            df = pd.read_csv(self.linkedfile)
            keepTracks = track_keeper(df, listTimes, self.srcDir, self.srcDir)
            keepTracks.update_DfTracks(0)
            dfagg = import_dfagg(self.aggFile)
            filTracks = keepTracks.filterAllTracks(dfagg)
            keepTracks.saveDfTracks()
            df_tracks = keepTracks.getDfTracks()
        else:
            df = pd.read_csv(self.linkedfile)
            df_tracks = pd.read_csv(self.dftracksfile)
            dfagg = import_dfagg(self.aggFile)
            listTimes = np.loadtxt(self.timesfile)
            filTracks = df_tracks[df_tracks.type==2].trackNr.values

        if not 'ews' in df.keys():
            df['ews'] = df.ew2 / df.ew1
        # Groupc
        dfg = groupdf(df)

        return df, df_tracks, dfagg, listTimes, filTracks


    def overlap(self, df, df_tracks, dfagg, listTimes, filTracks, listImgs):
        if (not isfile(self.dfoverlapfile)):
            dfoverlap = create_dfoverlap(filTracks, dfagg, dfg, self.dfoverlapfile)
        else:
            dfoverlap = pd.read_csv(self.dfoverlapfile)
        # Calculalte overlap
        filTracks = dfoverlap.trackNr.values
        filLengths = dfoverlap[['length1', 'length2']].values
        seg_functions = get_segFunctions(4, 3, (0.7, 3, 5), (0.85, 3, 5),
                        (0.7, 3, 0.8, 2.8, 6), (0.55, 2.6, 0.65, 2, 6))
        overlap = calcOverlap(listImgs,
                              listTimes,
                              df,
                              filTracks,
                              self.srcDir,
                              self.background,
                              getDist,
                              getIntDark,
                              seg_functions,
                              ofs=None,
                              darkphases=None,
                              plotAnimation=True,
                              plotImages=True,
                              filLengths=filLengths)

        notFilTracks = overlap.getUnsuccessfulTracks()
        df_tracks.loc[df_tracks.trackNr.isin(notFilTracks), 'type']=np.nan
        filTracks = df_tracks[df_tracks.type==2].trackNr.values
        df_tracks.to_csv(dftracksfile)









if __name__ == '__main__':
    argv = sys.argv
    ProcessExperiment(argv)
