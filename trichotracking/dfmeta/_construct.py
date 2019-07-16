from iofiles import getSubdirectories, find_txt

def getFiles(dfMeta, file):
    files = []
    expLabels = dfMeta.exp.values

    for exp in expLabels:
        dfExp = dfMeta[dfMeta.exp==exp]
        dir = dfExp.resultDir.values[0]
        keywords = dfExp.dirKeywords.values[0]
        resultDirs = getSubdirectories(dir, keywords)
        files.append(find_txt(dir, keywords, mustKw=file))
    return files


def getTrackFiles(dfMeta):
    file = 'tracks_linked.txt'
    files = getFiles(dfMeta, file)
    return files

def getTimeFiles(dfMeta):
    file = 'times.txt'
    files = getFiles(dfMeta, file)
    return files

def getDfaggFiles(dfMeta):
    file = 'tracks_agg.txt'
    files = getFiles(dfMeta, file)
    return files

def getDftrackFiles(dfMeta):
    file = 'df_tracks.txt'
    files = getFiles(dfMeta, file)
    return files
