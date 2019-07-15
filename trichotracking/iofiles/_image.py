import os
import os.path
import numpy as np
import cv2

from ._list_files import find_img

from IPython.core.debugger import set_trace


__all__ = ['getBackground', 'getChamber', 'loadImage']


def is_corrupted(file):
    """ Checks if given jpg file is corrupted. """
    with open(file, 'rb') as f:
        check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            print('Not complete image')
            return True
        else:
            return False


def loadImage(file, as_gray=True, as_8bit=True):
    """ Loads image from file and returns image, height and width.

    Parameters
    ----------
    file : string
        Image file name, e.g. ``test.jpg``
    as_gray: bool, optional
        If True, imports color images as gray-scale
    as_8bit: bool, optional
        If True, imports 16-bit image as 8-bit

    Returns
    -------
    img : ndarray
        If file is defect, returns -1
    height: int,
        Height of image. If file is defect, returns -1
    width: int
        Width of image. If file is defect, returns -1
    """

    if file2.lower().endswith('.jpg') and is_corrupted(file):
        return -1, -1,  -1


    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if as_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if as_8bit:
        img = np.uint8(cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX))
    height, width = img.shape[:2]


    return img, height, width







def getChamber(inputDir, background, chamber_function):
    """Import or calculate chamber image"""
    if "chamber.tif" in os.listdir(inputDir):
        chamber = loadImage(os.path.join(inputDir,"chamber.tif"))
    else:
        chamber = chamber_function(background)
        cv2.imwrite(os.path.join(inputDir, "chamber.tif"), chamber)
    return chamber
