import os
import os.path

__all__ = ['removeDirs', 'removeFiles', 'removeFilesinDir']


def removeFilesinDir(dir):
    filelist = [f for f in os.listdir(dir) if f.endswith(".tif")]
    for f in filelist:
        os.remove(os.path.join(dir, f))


def removeDirs(dirs):
    for dir in dirs:
        removeFilesinDir(dir)
        os.rmdir(dir)


def removeFiles(files):
    for file in files:
        os.remove(file)
