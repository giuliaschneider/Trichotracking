# -*- coding: utf-8 -*-

import os.path
import matplotlib as mpl


__all__ = ['saveplot']


def saveplot(fig, filename, saveDir):
    mpl.rcParams.update({'font.size': 10})
    #"""
    mpl.use("pgf", warn=False, force=True)

    pgf_with_custom_preamble = {
    "text.usetex": True,    # use inline math for ticks
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    #"pgf.rcfonts": False,   # don't setup fonts from rc parameters
    "pgf.preamble": [
         "\\usepackage{units}",
         "\\usepackage{unicode-math}",
         "\setmathfont{xits-math.otf}"
         ]

    }
    mpl.rcParams['text.latex.unicode'] = True
    mpl.rcParams.update(pgf_with_custom_preamble)
    #"""

    basename = os.path.basename(filename)
    if basename.endswith(".png"):
        basename = basename[:-4]

    if filename.endswith(".png"):
        filename = filename[:-4]

    fig.savefig(filename + ".pdf",bbox_inches='tight')


    #updateFig(fig, 'serif')
    file_str =  os.path.join(saveDir, basename)
    fig.savefig(file_str + ".pdf",bbox_inches='tight', dpi=300)
