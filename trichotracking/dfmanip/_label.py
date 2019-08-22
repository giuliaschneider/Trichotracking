__all__ = ['calcLabel']


def calcLabel(trackNr, chamber, exp):
    labels = []
    if isinstance(trackNr, int):
        return exp + str(chamber) + '{:04}'.format(trackNr)
    else:
        for i in trackNr:
            labels.append(exp + str(chamber) + '{:04}'.format(int(i)))
        return labels
