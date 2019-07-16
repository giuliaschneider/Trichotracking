def get_agginducer(dfMeta, exp):
    agginducer = dfMeta[dfMeta.exp==exp].agginducer.values[0]
    return agginducer

def get_chambers(dfMeta, exp):
    chambers = dfMeta[dfMeta.exp==exp].dirLabels.values[0]
    return chambers

def get_darkphases(dfMeta, exp):
    darkphases = dfMeta[dfMeta.exp==exp].darkphases.values[0]
    return darkphases

def get_px(dfMeta, exp):
    pxLength = dfMeta[dfMeta.exp==exp].pxLength.values[0]
    return pxLength


def get_aggchambers(dfMeta, exp):
    return dfMeta[dfMeta.exp==exp].aggchambers.values[0]
