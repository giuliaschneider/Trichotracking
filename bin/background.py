import sys

from trichotracking.iofiles import find_img
from trichotracking.segmentation import (getBackground,
                                         getChamber,
                                         calc_chamber_df_ulisetup,
                                         dilate_border,
                                         filterParticlesArea,
                                         particles_image)

dir = sys.argv[1]
bg = getBackground(dir)
chamber = getChamber(dir, bg, calc_chamber_df_ulisetup)
dchamber = dilate_border(chamber, ksize=800)

listImgs = find_img(dir)
particles_image(listImgs[0],
                0,
                bg,
                filterParticlesArea,
                chamber=dchamber,
                blur=True,
                darkField=True,
                plotImages=True,
                threshold=28)
