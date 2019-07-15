import os
import os.path
import numpy as np
import pandas as pd
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

from ._list_files import find_files
from ._metadata import getTime_timestamp
from ._image import loadImage
from geometry import cropRectangleKeepSize

from IPython.core.debugger import set_trace

__all__ = ['export_movie', 'export_movie_track']


def write_time(img, time, width, scale=1):
    # Write Time
    font                   = cv2.FONT_HERSHEY_DUPLEX
    bottomLeftCornerOfText = (width-100*scale,40*scale)
    fontScale              = scale
    fontColor              = (255)
    txt = time.strftime('%H:%M')
    cv2.putText(img, txt, bottomLeftCornerOfText,
        font, fontScale, fontColor)


def export_movie(dir, fps=20):
    """ Saves movie to parent directory of dir. """
    # Get files, directories
    listOfFiles = find_files(dir)
    #saveDir, _ = os.path.split(dir)
    saveDir, filename = os.path.split(dir)
    print(saveDir)
    print(filename)
    filename = os.path.join(saveDir, filename + '.avi')

    # Get image size
    img, height, width= loadImage(listOfFiles[0])
    size =  (img.shape[1] ,img.shape[0])

    # Create viedo writer
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    out = cv2.VideoWriter(filename, fourcc, fps, size, 0)

    for frame in range(0, len(listOfFiles)):
        # Write the frame into the file 'output.avi'
        img, height, width= loadImage(listOfFiles[frame])
        time = getTime_timestamp(listOfFiles[frame])
        write_time(img, time, width, scale=6)
        if img is None:
            set_trace()
        out.write(img)

    out.release()
    cv2.destroyAllWindows()


def export_movie_track(df, listOfFiles, listTimes, filename, fps, track,
                  dark=None, startFrame=None, endFrame=None):
    """ Saves movie to filename. """
    df = df[df.trackNr==track]
    bx = df.bx.min()
    by = df.by.min()
    bw = (df.bx + df.bw).max() - bx
    bh = (df.by + df.bh).max() - by
    mult = 1.1

    # Get image shape
    img, height, width= loadImage(listOfFiles[0])
    crop_params = bx, by, bw, bh, mult
    cropped, nbx, nby = cropRectangleKeepSize(img, *crop_params)
    size =  (cropped.shape[1] ,cropped.shape[0])

    times = np.array([datetime.fromtimestamp(i) for i in listTimes])

    if startFrame is None:
        startFrame = df.frame.values[0]

    if endFrame is None:
        endFrame = df.frame.values[-1]

    # Create viedo writer
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(filename, fourcc, fps, size, 0)


    for frame in range(startFrame, endFrame):
        print(frame)
        img, height, width= loadImage(listOfFiles[frame])
        crop_params = bx, by, bw, bh, mult
        cropped, nbx, nby = cropRectangleKeepSize(img, *crop_params)

        write_time(cropped, times[frame], bw)

        # Put point if light
        if((dark is not None) and (not dark[frame])):
            cv2.circle(cropped, (bw-50,60), 5, (255), -1)

        cropped = cropped.astype(np.uint8)
        size =  (cropped.shape[1] ,cropped.shape[0])

        # Write the frame into the file 'output.avi'
        out.write(cropped)

    out.release()
    cv2.destroyAllWindows()
