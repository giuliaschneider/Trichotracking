import os
import sys

from trichotracking.iofiles import find_img
from trichotracking.segmentation import (getBackground,
                                         getChamber,
                                         calc_chamber_df_ulisetup,
                                         dilate_border,
                                         filterParticlesArea,
                                         particles_image)

srcDir = sys.argv[1]
threshold = int(sys.argv[2])
dirs = sys.argv[3:]

for directory in dirs:
    directory = os.path.join(srcDir, directory)
    bg = getBackground(directory)
    chamber = getChamber(directory, bg, calc_chamber_df_ulisetup)
    dchamber = dilate_border(chamber, ksize=800)

    listImgs = find_img(directory)
    particles_image(listImgs[0],
                    0,
                    bg,
                    filterParticlesArea,
                    chamber=dchamber,
                    blur=True,
                    darkField=True,
                    plotImages=True,
                    threshold=threshold)
