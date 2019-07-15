import numpy as np
import pandas as pd
from dfmeta import get_files
from iofiles import find_img

from IPython.core.debugger import set_trace

def create_dfoverlap(filTracks, dfagg, dfg, dfoverlapfile):
    filTracks, filLengths, breakup = get_filLengths(filTracks, dfagg, dfg)
    filLengths.sort(axis=1)
    filLengths = filLengths[:,::-1]
    df = pd.DataFrame({'trackNr': filTracks,
                       'length1': filLengths[:,0],
                       'length2': filLengths[:,1],
                       'breakup': breakup,
                       'type': np.nan})
    df.to_csv(dfoverlapfile)
    return df


def get_input_overlap(dfMeta, exp, chamber):
    """ Get list of image, times and df files. """
    # Import all files of experiment
    trackFiles, aggFiles, timesFiles, dataDirs, resultDirs = \
            get_files(dfMeta,exp)

    # Get files of chamber
    i = int(chamber) - 1
    trackFile = trackFiles[i]
    aggFile = aggFiles[i]
    timesFile = timesFiles[i]
    dataDir = dataDirs[i]
    resultsDir = resultDirs[i]

    listOfImgs = find_img(dataDir)
    listTimes = times = np.loadtxt(timesFile)
    return (listOfImgs, listTimes, trackFile, aggFile, timesFile, dataDir,
            resultsDir)


def get_filLengths(filTracks, dfagg, dfg):
    """ Get aggregate trackNr & lenghts of single filaments. """

    # Merge dfagg and dfg
    vdfg = dfg[['trackNr', 'length_mean']] # view of dfg
    props = {'how':'left', 'right_on': 'trackNr'}
    suf = ('0','1')
    dfagg = pd.merge(dfagg,vdfg,**props,left_on='deftrack1',suffixes=suf)
    suf = ('1','2')
    dfagg = pd.merge(dfagg,vdfg,**props,left_on='deftrack2',suffixes=suf)

    cols = ['trackNr0', 'length_mean1', 'length_mean2', 'breakup']
    dftracks = dfagg[dfagg.trackNr0.isin(filTracks)][cols]
    tracks = dftracks.trackNr0.values
    filLengths = dftracks[['length_mean1', 'length_mean2']].values
    breakup = dftracks.breakup.values
    return tracks, filLengths, breakup
