import cv2
import numpy as np

__all__ = ['loadImage']


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

    if file.lower().endswith('.jpg') and is_corrupted(file):
        return -1, -1, -1

    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if (img.ndim > 2) and as_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if as_8bit:
        img = np.uint8(cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX))
    height, width = img.shape[:2]

    return img, height, width
