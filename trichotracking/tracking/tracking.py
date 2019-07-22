import sys
import os
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from iofiles import (find_img,
                     getTime,
                     loadImage)
from linking import matcher
from plot.plot_images import plotAllContoursTracking

from ._seg_functions import segment_particles

from IPython.core.debugger import set_trace



class tracker:
    def __init__(self,
                 input_dir,
                 output_dir,
                 pxLength,
                 maxR,
                 findingFunction,
                 background,
                 linkTime=1,
                 minLength=1,
                 plotImages=False,
                 threshold=28,
                 roi=None,
                 blur=True,
                 darkField=False):

        """ Tracks particles in the video given in input_dir.

        Parameters
        ----------
        input_dir --  absolute/relative path to dir containg images
        output_dir -- absolute/relative path to output directory
        pxLength -- pixel-length conversion in µn/px
        maxR --  max distance (µm) a particle can move between frames
        minLength --minimal track length
        background -- background image
        plotContours -- bool indicating if Contours should be plotted
        """

        # Initalize
        self.listOfFiles = find_img(input_dir)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.pxLength = pxLength
        self.maxDist = 1.0*maxR/self.pxLength
        self.findingFunction = findingFunction
        self.background = background
        self.plotImages = plotImages
        self.threshold = threshold
        self.roi = roi
        self.blur = blur
        self.darkField = darkField

        self.nFiles = len(self.listOfFiles)
        self.height, self.width = self.background.shape[:2]

        self.track(linkTime)

        print("Done Tracking")



    def track(self, linkTime):
        """ Tracks particles, saves results to dataframe ParticleTracks.

        Particles are matched to previous particles by distance and
        assign he right track nr.

        Keyword arguments:
        linkTime:   Missing data is filled by going back linkTime
                    number of time steps(time.ctime
                    If linkTime == 1: only previous time step is considered

        """
        # check linkTime
        if linkTime < 1:
            linkTime = 1

        # initialize variables
        nTracks = 0
        self.particleList = []  # list of dictionary saving all tracks
        cxPrev = []             # list saving the past linkTime x coord
        cyPrev = []             # list saving the past linkTime y coord
        trackNrPrev = []        # list saving the past linkTime track nr
        self.imgTimes = []


        # iterate images
        for frame, file in enumerate(self.listOfFiles):

            # segmentation
            self.img, _, _ = loadImage(file)
            if (isinstance(self.img, int)) and (self.img == -1):
                continue

            self.img, bw = segment_particles(self.img, 
                                             self.background, 
                                             chamber=self.roi,
                                             blur=self.blur, 
                                             darkField=self.darkField, 
                                             plotImages=self.plotImages, 
                                             threshold=self.threshold)

            self.imgTimes.append(getTime(file))
            self.particles = self.findingFunction(self.img, bw, self.roi)

            if self.plotImages:
                plotAllContoursTracking(self.img, self.particles.contours,
                                self.particles.cx, self.particles.cy)
                plt.show()

            # number of current particles
            nCurrent = self.particles.shape[0]
            # Current particle positions
            cxCur = np.array(self.particles.cx)
            cyCur = np.array(self.particles.cy)

            print("Track Frame = {}, {} Particles".format(frame, nCurrent))

            if frame == 0: # first image
                # add all particles to particle list
                trackNrCur = np.arange(0,nCurrent)
                self.updateParticleList(trackNrCur, frame, trackNrCur)

                # update nr of Tracks
                nTracks += nCurrent

            else: # Other images
                trackNrCur = np.zeros(cxCur.shape) - 1

                # iterate through all linkTime previous frames,
                # starting with last time step
                for t in range(len(cxPrev)):
                    # get not matched particles
                    # oIndNMC = original index not matching current
                    oIndNMC = np.where(trackNrCur < 0)[0]

                    if oIndNMC.size > 0:
                        cxNMCur = cxCur[oIndNMC]
                        cyNMCur = cyCur[oIndNMC]

                        # Matching particles of current with previous time step
                        # indMP = index matching previous
                        cxP = cxPrev[t]
                        cyP = cyPrev[t]
                        trackNrP = trackNrPrev[t]
                        boolAlreadyM = np.in1d(trackNrP, trackNrCur)
                        cxP = cxP[~boolAlreadyM]
                        cyP = cyP[~boolAlreadyM]
                        trackNrP = trackNrP[~boolAlreadyM]

                        indMP, indMC = matcher(cxP, cyP, cxNMCur, cyNMCur,
                                             self.maxDist*(t+1))

                        # print("Matched  = {}".format(len(indMP)))

                        # get original index of matched current particle
                        oIndMC = oIndNMC[indMC]

                        # array of trackNr of previous matched particles
                        trackNrMP= trackNrP[indMP]
                        trackNrCur[oIndMC] = trackNrMP

                        # Add matched particles to list
                        self.updateParticleList(oIndMC, frame, trackNrMP)

                        #if t > 0 and trackNrMP.size > 0:
                            # Fill data for missing frames
                            #self.fillData(t+1, frame, trackNrMP, cxP[indMP],
                            #    cyP[indMP],
                            #    cxNMCur[indMC],
                            #    cyNMCur[indMC],
                            #    oIndMC)

                # add new tracks to particle list
                indNMC = np.where(trackNrCur < 0)[0]
                newTrackNr = np.arange(nTracks,nTracks+len(indNMC))
                trackNrCur[indNMC] = newTrackNr
                self.updateParticleList(indNMC, frame, newTrackNr)

                # Update running variables
                nTracks += len(indNMC)

            # Update list of previous particles
            cxPrev.insert(0, cxCur)
            cyPrev.insert(0, cyCur)
            trackNrPrev.insert(0, trackNrCur)
            if len(cxPrev) > linkTime:
                cxPrev.pop()
                cyPrev.pop()
                trackNrPrev.pop()
            del self.img
            plt.show()

        # Convert list dictionary to dataframe
        self.particleTracks = pd.DataFrame(self.particleList)
        self.nTracks = nTracks
        print("nTracks = {}".format(self.nTracks))



    def getParticleList(self, vCurrentIndex, frame, vTrackNr):
        """ Saves particles in list of dictionaries (performance). """
        particleList = []
        for index, trackNr in zip(vCurrentIndex, vTrackNr):
            partDict = self.particles.iloc[index].to_dict()
            partDict["frame"] = frame
            partDict["trackNr"] = trackNr
            particleList.append(partDict)
        return particleList


    def updateParticleList(self, vCurrentIndex, frame, vTrackNr):
        """ Adds the new particles to the particle list."""
        newList = self.getParticleList(vCurrentIndex, frame, vTrackNr)
        self.particleList += newList


    def fillData(self, t, frame, vTrackNr, cxP, cyP, cxC, cyC, vCurrentIndex):
        """ Fills missing data points if particle is missing in frame.

        Keyword arguments:
        t --    number of backward time steps
        frame -- current frame nr
        vTrackNr np array of particle track Nr
        cxP --   x coord. of particles of previous time, np.array
        cyP --   y coord. of particles of previous time, np.array
        cxC --   x coord. of particles of previous time, np.array
        cyC --   y coord. of particles of previous time, np.array
        vCurrentIndex -- index of current particle in self.particles

        """

        # iterate through previous time steps
        #print("Filling in Data, Frame = {}".format(frame))
        particleList = []

        for i in range(1,t):
            #print("i = {}".format(i))
            # calculate the missing position as weighted average
            weigthC = t-i
            weightP = t-weigthC
            cxM = (weigthC*cxC + weightP*cxP)/t
            cyM = (weigthC*cyC + weightP*cyP)/t

            # save generated data in particle list
            index = np.arange(cxM.size)
            for j, origInd, trackNr in zip(index, vCurrentIndex, vTrackNr):
                partDict = self.particles.iloc[origInd].to_dict()
                #partDict = {}
                partDict["frame"] = frame-i
                partDict["trackNr"] = trackNr
                partDict["cx"] = cxM[j]
                partDict["cy"] = cyM[j]
                particleList.append(partDict)

        self.particleList += particleList


    def getParticleTracks(self):
        return self.particleTracks

    def getTimes(self):
        return self.imgTimes

    def getListOfImgs(self):
        return self.listOfFiles
