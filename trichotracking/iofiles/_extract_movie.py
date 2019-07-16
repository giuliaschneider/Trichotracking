import os
import os.path
import numpy as np
import pandas as pd
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

from ._list_files import find_files, find_img
from ._metadata import getTime_timestamp
from ._image import loadImage
from geometry import cropRectangleKeepSize

from IPython.core.debugger import set_trace

__all__ = ['export_movie', 'export_movie_track']


def write_particles(img, dfparticles, frame):
        cx = dfparticles[dfparticles.frame == frame].cx
        cy = dfparticles[dfparticles.frame == frame].cy
        trackNr = dfparticles[dfparticles.frame == frame].trackNr   
        for ccx, ccy, nr in zip(cx, cy, trackNr):
            ccx = int(ccx)
            ccy = int(ccy)   
            cv2.circle(img, (ccx, ccy), (3), (255, 255, 255))
            font                   = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            bottomLeftCornerOfText = (ccx+10,ccy+10)
            fontScale              = 2
            fontColor              = (255, 255, 255)
            txt = "{:.0f}".format(nr)
            cv2.putText(img, txt, bottomLeftCornerOfText,
                font, fontScale, fontColor)
            



def write_time(img, time, width, scale=1):
    # Write Time
    font                   = cv2.FONT_HERSHEY_DUPLEX
    bottomLeftCornerOfText = (width-100*scale,40*scale)
    fontScale              = scale
    fontColor              = (255)
    txt = time.strftime('%H:%M')
    cv2.putText(img, txt, bottomLeftCornerOfText,
        font, fontScale, fontColor)


def export_movie(dir, fps=20, dfparticles=None, nTracks=None,
                 filename=None):
    """ Saves movie to parent directory of dir. """
    # Get files, directories
    listOfFiles = find_img(dir)

    filename = os.path.join(dir,  'animation.avi')

    # Get image size
    img = loadImage(listOfFiles[0])[0]
    size =  (img.shape[1] ,img.shape[0])

    # Create viedo writer
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    out = cv2.VideoWriter(filename, fourcc, fps, size, 0)

    for frame in range(0, len(listOfFiles)):
        # Write the frame into the file 'output.avi'
        img, _, width = loadImage(listOfFiles[frame])
        if isinstance(img, int) and (img == -1):
            continue
        time = getTime_timestamp(listOfFiles[frame])
        write_time(img, time, width, scale=6)
        if dfparticles is not None:
            write_particles(img, dfparticles, frame)
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
    img = loadImage(listOfFiles[0])[0]
    crop_params = bx, by, bw, bh, mult
    cropped = cropRectangleKeepSize(img, *crop_params)[0]
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
        img = loadImage(listOfFiles[frame])[0]
        crop_params = bx, by, bw, bh, mult
        cropped  = cropRectangleKeepSize(img, *crop_params)[0]

        write_time(cropped, times[frame], bw)

        # Put point if light
        if((dark is not None) and (not dark[frame])):
            cv2.circle(cropped, (bw-50,60), 5, (255), -1)

        size =  (cropped.shape[1] ,cropped.shape[0])

        # Write the frame into the file 'output.avi'
        out.write(cropped)

    out.release()
    cv2.destroyAllWindows()
