from iofiles import getSubdirectories, find_txt


def get_files(dfMeta, exp):
    trackFiles = dfMeta[dfMeta.exp==exp].trackFiles.values[0]
    timesFiles = dfMeta[dfMeta.exp==exp].timesFiles.values[0]
    aggFiles = dfMeta[dfMeta.exp==exp].dfaggFiles.values[0]
    dftrackFiles = dfMeta[dfMeta.exp==exp].dftrackFiles.values[0]
    return trackFiles, aggFiles, dftrackFiles, timesFiles

def get_metadfo(dfMeta, exp):
    dataDir, dataDirs, resultDir, resultDirs = get_dirs(dfMeta, exp)
    metadfo = find_txt(resultDir, 'df_overlap')
    return metadfo

def get_dirs(dfMeta, exp):
    resultDir = dfMeta[dfMeta.exp==exp].resultDir.values[0]
    dataDir = dfMeta[dfMeta.exp==exp].dataDir.values[0]
    keywords = dfMeta[dfMeta.exp==exp].dirKeywords.values[0]
    dataDirs = getSubdirectories(dataDir, keywords)
    resultDirs = getSubdirectories(resultDir, keywords)
    return dataDir, dataDirs, resultDir, resultDirs
