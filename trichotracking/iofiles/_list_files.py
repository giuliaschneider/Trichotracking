import os
import os.path

__all__ = ['find_files', 'find_img', 'find_txt', 'getSubdirectories']


def find_files(dir, keywords=None, mustKw=None):
    """ Returns a  sorted list of files in dir containing keywords. """
    allFiles = os.listdir(dir)
    if mustKw is not None:
        allFiles = [i for i in allFiles if mustKw in i.lower()]

    if keywords is None:
        files = [os.path.join(dir, i) for i in allFiles]
    elif isinstance(keywords, str):
        files = [os.path.join(dir, i) for i in allFiles if keywords in i]
    else:
        files = []
        for keyword in keywords:
            temp = [os.path.join(dir, i) for i in allFiles if keyword in i]
            files += temp
    files.sort()
    return files


def find_img(dir, keywords=None, mustKw='.tif'):
    """ Returns a sorted list of all image files in path. """
    temp = find_files(dir, keywords, mustKw)
    if len(temp) < 3:
        temp = find_files(dir, keywords, '.jpg')
    files = [f for f in temp if "background" not in f and "chamber.tif" not in f]
    return files


def find_txt(dir, keywords=None, mustKw='.txt'):
    """ Returns a sorted list of all text files in path. """
    files = find_files(dir, keywords, mustKw)
    return files


def getSubdirectories(dirPath, keywords=None):
    """ Returns sorted list of all the absolute path of all subdirs"""
    subdirs = [name for name in os.listdir(dirPath)
               if os.path.isdir(os.path.join(dirPath, name))]
    if keywords is None:
        abs_subdirs = [os.path.join(dirPath, i) for i in subdirs]
    elif isinstance(keywords, str):
        abs_subdirs = [os.path.join(dirPath, i) for i in subdirs if keywords in i]
    else:
        abs_subdirs = []
        for keyword in keywords:
            abs_subdirs += [os.path.join(dirPath, i) for i in subdirs if keyword in i]

    abs_subdirs.sort()
    return abs_subdirs
