# -*- coding: utf-8 -*-

import os.path
import matplotlib as mpl

reportDir = "/home/giu/Documents/10Masterarbeit/report/Master-Thesis/draft/Figures"
presentationDir = "/home/giu/Documents/10Masterarbeit/presentation/Figures"


def updateFig(fig, font):
    ax = fig.gca()
    items = ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels() +
             ax.legend().get_texts())
    for item in items:
        item.set_fontname(font)



def saveplot(fig, filename):
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
    file_str =  os.path.join(reportDir, basename)
    fig.savefig(file_str + ".pdf",bbox_inches='tight', dpi=300)

    #updateFig(fig, 'sans-serif')
    #file_str =  os.path.join(presentationDir, basename)
    #fig.savefig(file_str + ".pdf",bbox_inches='tight')
