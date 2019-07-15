from PIL import Image
from PIL.TiffTags import TAGS
import exifread
import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

"""plt.switch_backend("pgf")
pgf_with_custom_preamble = {
    "text.usetex": True,    # use inline math for ticks
    "font.family": "serif",
    "font.serif": [],
}
mpl.rcParams.update(pgf_with_custom_preamble)

mpl.rcParams.update({'font.size': 12})"""

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.colors import ListedColormap

def get_mycmap():
    mycmap = np.zeros((4,4))
    mycmap = [cm.pink(i/4) for i in range(4)]
    new_cmp = ListedColormap(mycmap)
    return new_cmp



def moveImage(fig, x, y):
    """ Moves figure to window position (x,y). """
    backend = mpl.get_backend()
    if backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        fig.canvas.manager.window.SetPosition((x, y))

def get_exif(fn):
    print(fn)
    print(time.ctime(os.path.getmtime(fn)))
    f = open(fn, 'rb')

    # Return Exif tags
    tags = exifread.process_file(f)

    # Print the tag/ value pairs
    for tag in tags.keys():
        if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
            print("Key: {}, value {}".format(tag, tags[tag]))

def get_hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    #hist, bins = np.histogram(img, 100)
    plt.figure()
    plt.plot(hist)
    #hist[np.where(hist<0.01*np.max(hist))] = 0

    maxX = np.max(np.nonzero(hist))
    #plt.xlim([0,maxX])
    plt.ylim([0, np.max(hist[1:])])

def get_hist_non_img(img):
    hist, bins = np.histogram(img, 100)
    plt.figure()
    plt.plot(bins[:-1], hist)



def removeNoise(bw, minArea):
    """ Removes all objects smaller than minArea. """
    nObjects, labelledImage, stats, centroids = \
        cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
    objectAreas = stats[:,cv2.CC_STAT_AREA]
    labels = np.arange(nObjects)
    noiseLabels = labels[objectAreas<minArea]
    for i in noiseLabels:
        bw[labelledImage==i] = 0
    return bw



def save_image_fix_dpi(img, figname, cmap, c, cbTitle, vmin=None,
                       vmax=None, dpi=75, ticks=None, tickLabels=None):
    shape=np.shape(img)[0:2][::-1]
    size = [float(i)/dpi for i in shape]
    print(size)

    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig,[0,0,1,1])
    #fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    fig.add_axes(ax)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

    # COLORBAR
    axins1 = inset_axes(ax, width="75%",height="4%",loc='lower right',
                        borderpad=0.75)
    cb = plt.colorbar(im, cax=axins1, orientation='horizontal', pad=0.05)
    # set colorbar label plus label color
    #cb.set_label(cbTitle, color=c)
    axins1.xaxis.set_label_position("top");
    # set colorbar tick color
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels(tickLabels)
    cb.ax.xaxis.set_tick_params(color=c)
    # set colorbar edgecolor
    cb.outline.set_edgecolor(c)
    # set colorbar ticklabels
    axins1.xaxis.set_ticks_position("top");
    plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=c)

    ax.set_axis_off()

    fig.savefig(figname + ".png",dpi=fig.dpi, bbox_inches='tight')
    fig.savefig(figname + ".pdf",dpi=fig.dpi, bbox_inches='tight')
