



def set_plotLabel(row, dfMeta):
    exp = row.name[0]
    dark = row.name[1]
    if dark==1:
        return dfMeta[dfMeta.exp==exp].labelDark.values[0]
    else:
        return dfMeta[dfMeta.exp==exp].labelLight.values[0]
