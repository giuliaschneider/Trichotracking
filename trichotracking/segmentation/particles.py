import multiprocessing as mp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trichotracking.iofiles import (find_img,
                                    loadImage)
from trichotracking.plot import plotAllImages, plotAllContoursTracking
from trichotracking.regionprops import Contour, insideROI

flags = ["area", "angle", "bounding_box", "centroid", "contours", "eccentricity",
         "eigen", "length", "min_box"]


def filterParticlesArea(img, bw, roi=None):
    allObjects = Contour(img, bw, flags=flags, cornerImage=None)
    particles = allObjects.particles
    particles = particles[((particles.area > 50)
                           & (particles.area < 5000)
                           )]
    if roi is not None:
        inside = insideROI(particles.min_box.values, roi)
        particles = particles[inside]
    return particles


def getParticleList(particles, vCurrentIndex, frame, vTrackNr):
    """ Saves particles in list of dictionaries (performance). """
    particleList = []
    for index, trackNr in zip(vCurrentIndex, vTrackNr):
        partDict = particles.iloc[index].to_dict()
        partDict["frame"] = frame
        partDict["index"] = trackNr
        partDict["trackNr"] = trackNr
        particleList.append(partDict)
    return particleList


def particles_image(file,
                    frame,
                    background,
                    findingFunction,
                    chamber=None,
                    blur=True,
                    darkField=False,
                    plotImages=False,
                    threshold=28):
    """ Segments image and returns property list of found particles. """

    # Segment image(file)
    img = loadImage(file)[0]
    if blur:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    if darkField:
        subtracted = cv2.subtract(img, background)
        _, bw = cv2.threshold(subtracted, threshold, 255, cv2.THRESH_BINARY)
    else:
        subtracted = cv2.subtract(cv2.bitwise_not(img),
                                  cv2.bitwise_not(background))
        _, bw = cv2.threshold(subtracted, threshold, 255, cv2.THRESH_BINARY)

    if chamber is not None:
        bw[chamber == 0] = [0]

    # Close gaps by morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    # Get particles
    particles = findingFunction(img, bw, chamber)
    nCurrent = particles.shape[0]
    print("Track Frame = {}, {} Particles".format(frame, nCurrent))
    particleNr = np.arange(0, nCurrent) + frame * 1000
    indexes = np.arange(0, nCurrent)
    particleList = getParticleList(particles, indexes, frame, particleNr)

    # plot images
    if plotImages:
        imgs = [img, bw, subtracted]
        labels = ["Original", "Threshold", "Sub"]
        plotAllImages(imgs, labels)
        plotAllContoursTracking(img, particles.contours, particles.cx, particles.cy)
        plt.show()

    return particleList


def particles_sequence(input_dir,
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
    listOfFiles = find_img(input_dir)
    nFrames = len(listOfFiles)
    listOfFrames = list(range(nFrames))

    pool = mp.Pool(processes=4)
    results = [pool.apply_async(particles_image,
                                args=(
                                    listOfFiles[x],
                                    listOfFrames[x],
                                    background,
                                    findingFunction,
                                    roi,
                                    blur,
                                    darkField,
                                    plotImages,
                                    threshold)) for x in range(nFrames)]

    pool.close()
    pool.join()

    particleList = []
    for result in results:
        particleList += result.get()

    return pd.DataFrame(particleList)
