

def calcLabel(trackNr, chamber, exp):
    labels = []
    if isinstance(trackNr, int):
        return  exp+str(chamber)+'{:04}'.format(trackNr)
    else:
        for i in trackNr:
            labels.append(exp+str(chamber)+'{:04}'.format(i))
        return labels

def set_plotLabel(row, dfMeta):
    exp = row.name[0]
    dark = row.name[1]
    if dark==1:
        return dfMeta[dfMeta.exp==exp].labelDark.values[0]
    else:
        return dfMeta[dfMeta.exp==exp].labelLight.values[0]
