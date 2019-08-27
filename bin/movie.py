import sys

from trichotracking.iofiles import export_movie

dir = sys.argv[1]
export_movie(dir, filestep=10, fps=7)
