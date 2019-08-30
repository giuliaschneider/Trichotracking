import os
import sys

from trichotracking.iofiles import export_movie

srcDir = sys.argv[1]
dirs = sys.argv[2:]

for directory in dirs:
    export_movie(os.path.join(srcDir, directory), filestep=10, fps=7)
