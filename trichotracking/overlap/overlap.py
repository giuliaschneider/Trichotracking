import os
import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trichotracking.geometry import cropRectangleKeepSize
from trichotracking.iofiles import (find_img,
                                    loadImage,
                                    removeFilesinDir)
from trichotracking.regionprops import getContours
from trichotracking.utility import meanOfList
from ._segment_overlap import SegmentOverlap

PARAMS_CONTOURS = (cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


class calcOverlap():
    """ Calculates the overlap from intensity weighted disance transform.

    Keyword arguments:
    listOfFiles --  list of all images
    listTimes --    list of times at which images where taken
    df_tracks --    dataframe of particle tracks
    tracks --       list of trackNr which are considered
    saveDir --      directory to which results are saved
    background --   background image during imaging
    previous_overlap_fraction -- overlap of shorter filament in previous frame
    plotAnimation --bool if animation of overlap should be saved
    plotImages --   bool if intermediate results are shown
    filLengths --   numpy array of single filament lengths
    """

    def __init__(self,
                 listOfFiles,
                 list_times,
                 pairkeeper,
                 df_tracks,
                 saveDir,
                 background,
                 getDistFunc,
                 getIntFunc,
                 seg_functions,
                 ofs=None,
                 darkphases=None,
                 plotAnimation=False,
                 plotImages=False):

        # Initalize variables
        self.listOfFiles = listOfFiles
        self.list_times = list_times
        self.pairkeeper = pairkeeper
        self.df_tracks = df_tracks
        self.df_tracks = df_tracks

        self.tracks = self.pairkeeper.getTrackNr()
        self.nTracks = len(self.tracks)
        self.filLengths = self.pairkeeper.getLengths()

        self.saveDir = saveDir
        if not os.path.isdir(self.saveDir):
            os.mkdir(self.saveDir)
        self.background = self.getBackground(background)
        self.getDistFunc = getDistFunc
        self.getIntFunc = getIntFunc
        self.seg_functions = seg_functions
        self.ofs = self.getOfs(ofs)
        self.darkphases = darkphases
        self.plotAnimation = plotAnimation
        self.plotImages = plotImages

        self.success = []
        self.failFraction = []

        self.df_list = []
        self.iterate_tracks()

        self.df = pd.concat(self.df_list)
        self.df.to_csv(os.path.join(self.saveDir, 'tracks_pair.csv'))
        self.pairkeeper.df['couldSegment'] = self.success
        self.pairkeeper.df['failFraction'] = self.failFraction

    def iterate_tracks(self):
        """ Calculates the overlap of each track and save results to txt."""

        # Iterate through all tracks
        for i, track in enumerate(self.tracks[:]):
            self.getBasename(track)

            if not os.path.isfile(self.basename + '.txt'):
                # Filter dataframe for current track
                df = self.df_tracks[self.df_tracks.trackNr == track]
                df = df.sort_values(by=['frame'])

                if self.plotAnimation:
                    self.getSaveDirectories(track)

                # Calculate overlap series for track
                self.iterate_frames(i, df, track)

                self.createDataFrame(track)
                success, nNan, nFrames = self.checkTrackSuccess()
                self.success.append(success)
                self.failFraction.append(nNan / nFrames)
                print("Track Nr {} segmented, {} Nan values of total {} frames, ".format(track, nNan, nFrames))
                if success:
                    self.saveDataFrame(track)
                    if self.plotAnimation:
                        self.saveAnimation(track)
                else:
                    if self.plotAnimation:
                        removeFilesinDir(self.saveDir_img)
                        removeFilesinDir(self.saveDir_bw)
                        os.rmdir(self.saveDir_img)
                        os.rmdir(self.saveDir_bw)
            else:
                self.success.append(True)

    def iterate_frames(self, i, df, track):
        """ Iterates through frames in single track and calculates overlap."""
        # Initiazlize lists
        self.lengths_filament1, self.lengths_filament2 = [], []
        self.cx1, self.cy1, self.cx2, self.cy2 = [], [], [], []
        self.lengths_overlap, self.xlovs, self.ylovs = [], [], []
        self.dirx1, self.diry1 = [], []
        self.short_fil_pos = []
        self.track_times = []

        # Initalize variables
        bw = df.bw.max()
        bh = df.bh.max()
        mult = 1.75
        self.avg_overlap = None
        self.previous_of = self.ofs[i]
        self.previous_seg = dict({
            "segmented": None,
            "overlap_fraction": self.previous_of,
            "cxcy": None,
            "labels": None,
            "long_fil": None,
            "short_fil": None})
        if self.filLengths is None:
            self.avg = None
        else:
            avg = self.filLengths[i, :]
            self.avg = avg[np.argsort(avg)[::-1]]

        if self.plotAnimation:
            removeFilesinDir(self.saveDir_img)
            removeFilesinDir(self.saveDir_bw)

        # Iterate through all track frames
        for frame in df.frame:
            df_frame = df[df.frame == frame]
            self.track_times.append(frame)
            self.frame = frame

            # Load and crop original image
            img = loadImage(self.listOfFiles[frame])[0]
            crop_params = df_frame.bx, df_frame.by, bw, bh, mult
            self.cropped, nbx, nby = cropRectangleKeepSize(img, *crop_params)
            self.cropped_bg, *_ = cropRectangleKeepSize(self.background,
                                                        *crop_params)

            self.nBx = nbx
            self.nBy = nby

            # Extract pixellist and update pixellist indexes
            contour = df_frame.contours.values[0]

            cropped_contour = contour - np.tile(np.array([nbx + 1, nby + 1]), (contour.shape[0], 1, 1))
            self.cropped_bw = np.zeros(self.cropped.shape[0:2], np.uint8)
            cv2.drawContours(self.cropped_bw, [cropped_contour], 0, (255), -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            self.cropped_bw = cv2.morphologyEx(self.cropped_bw, cv2.MORPH_CLOSE, kernel)

            # Calculate overlap
            self.calcOverlapFromShapeIntensity(track, frame)

    def calcOverlapFromShapeIntensity(self, track, frame):
        """ Calculates overlap for a single frame. """
        # Initialize variables
        img = self.cropped
        bw = self.cropped_bw
        bg = self.cropped_bg
        R = max(img.shape)
        if (self.avg_overlap is None) or np.isnan(self.avg_overlap):
            connectingR = R / 1.5
        else:
            connectingR = self.avg_overlap * 2.5

        # Remove noise and fill contours
        c_bw = getContours(bw)
        bw_filled = cv2.drawContours(bw, c_bw, -1, (255), -1)

        segment = SegmentOverlap(
            img, bw, bw_filled, bg, self.previous_seg, self.frame,
            self.avg, connectingR, self.nBx, self.nBy, self.getDistFunc,
            self.getIntFunc, self.seg_functions)

        # Save results to list
        self.update(track, frame, segment)

        if self.plotImages:
            fig = plt.figure(figsize=(8, 8))
            (x, y) = (600, 0)
            ax1 = fig.add_subplot(221)
            ax1.imshow(img)
            ax2 = fig.add_subplot(222)
            ax2.imshow(segment.dist)
            ax3 = fig.add_subplot(223)
            ax3.imshow(segment.int)
            ax4 = fig.add_subplot(224)
            im = ax4.imshow(self.previous_segented)
            plt.colorbar(im)
            plt.show()
        plt.close('all')

    def update(self, track, frame, segment):
        """ Adds results to result lists, updates input variables. """

        # Get lengths
        self.length_filaments, self.length_overlap = segment.getAllLengths()
        self.lengths_filament1.append(self.length_filaments[0])
        self.lengths_filament2.append(self.length_filaments[1])
        self.lengths_overlap.append(self.length_overlap)

        if self.filLengths is None:
            cum_lengths = self.length_filaments
            ind = 1
            self.avg = cum_lengths / ind
        self.avg_overlap = meanOfList(self.lengths_overlap[-3:])

        # Get lack of overlap
        self.xlov, self.ylov = segment.calcLackOfOverlap()
        self.xlovs.append(self.xlov)
        self.ylovs.append(self.ylov)

        # Get position
        self.pos = segment.calcShortFilPosition()
        self.short_fil_pos.append(self.pos)
        # Get filaments
        self.previous_seg = segment.getSegmentationProps()
        self.previous_segented = self.previous_seg["segmented"]
        if self.previous_seg['long_fil'] is None:
            self.cx1.append(np.nan)
            self.cy1.append(np.nan)
            self.dirx1.append(np.nan)
            self.diry1.append(np.nan)
        else:
            self.cx1.append(self.previous_seg['long_fil']['centroid'][0][0])
            self.cy1.append(self.previous_seg['long_fil']['centroid'][0][1])
            self.dirx1.append(self.previous_seg['long_fil']['direction'][0])
            self.diry1.append(self.previous_seg['long_fil']['direction'][1])

        if self.previous_seg['short_fil'] is None:
            self.cx2.append(np.nan)
            self.cy2.append(np.nan)
        else:
            self.cx2.append(self.previous_seg['short_fil']['centroid'][0][0])
            self.cy2.append(self.previous_seg['short_fil']['centroid'][0][1])

        if self.plotAnimation:
            basename = "track_{}_frame_{:03d}_".format(track, frame)
            figname = os.path.join(self.saveDir_img, basename + "_cropped.tif")
            _ = cv2.imwrite(figname, self.cropped)
            figname = os.path.join(self.saveDir_bw, basename + "_sep.tif")
            _ = cv2.imwrite(figname, np.uint8(self.previous_segented * 85))

    def createDataFrame(self, track):
        """ Create dataframe"""

        self.df_track = pd.DataFrame({
            "length1": self.lengths_filament1,
            "length2": self.lengths_filament2,
            "cx1": self.cx1,
            "cy1": self.cy1,
            "cx2": self.cx2,
            "cy2": self.cy2,
            "dirx1": self.dirx1,
            "diry1": self.diry1,
            "length_overlap": self.lengths_overlap,
            "xlol": self.xlovs,
            "ylol": self.ylovs,
            "pos_rel": self.short_fil_pos,
            "frame": self.track_times,
            "trackNr": track})

    def checkTrackSuccess(self):
        nanValues = ((self.df_track.length1.isnull())
                     | (self.df_track.length2 == 0))
        nNan = nanValues[nanValues].size
        nAll = nanValues.size
        self.df_track['block'] = (nanValues != nanValues.shift()).cumsum()
        nNotNan = self.df_track.groupby(by='block').length1.count().max()
        success = (nNan / nAll < 0.3) & ((nNotNan) > 5)
        return success, nNan, nAll

    def getBasename(self, track):
        name = "overlap_track_{}".format(track)
        self.basename = os.path.join(self.saveDir, name)

    def saveDataFrame(self, track):
        """ Save results to text file"""
        self.df_list.append(self.df_track)

    def saveAnimation(self, track):
        """Create animation and save as avi. """
        list_img = find_img(self.saveDir_img)
        list_bw = find_img(self.saveDir_bw)
        # ani = OverlapAnimation(list_img, list_bw, self.df_track, self.darkphases)
        # ani.save(self.basename + '.avi')
        plt.close('all')

    def getBackground(self, background):
        """ Checks if background is not None"""
        if background is None:
            img = loadImage(self.listOfFiles[0])[0]
            bg = np.zeros(img.shape[:2]).astype(np.uint8)
        else:
            bg = background
        return bg

    def getOfs(self, ofs):
        """ Checks if ofs is not None"""
        if ofs is None:
            ofs = [None for i in range(self.nTracks)]
        return ofs

    def getSaveDirectories(self, track):
        """ Creates directories to save bw/segmented images. """
        self.saveDir_img = os.path.join(self.saveDir,
                                        "cropped_track_{}".format(track))
        self.saveDir_bw = os.path.join(self.saveDir,
                                       "separated_track_{}".format(track))
        if os.path.isdir(self.saveDir_img):
            removeFilesinDir(self.saveDir_img)
        else:
            os.mkdir(self.saveDir_img)
        if os.path.isdir(self.saveDir_bw):
            removeFilesinDir(self.saveDir_bw)
        else:
            os.mkdir(self.saveDir_bw)

    def getUnsuccessfulTracks(self):
        success = np.array(self.success)
        if self.nTracks > 0:
            tracks = np.array(self.tracks)
            unsuccssful = tracks[~success]
        else:
            unsuccssful = []
        return unsuccssful

    def getDf(self):
        return self.df
