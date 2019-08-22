__all__ = ['meanOfList', 'split_list']


def meanOfList(inputList):
    """ Calculates the mean value of inputList."""
    if not inputList:
        return None
    else:
        return float(sum(inputList)) / len(inputList)


def chunks(inputList, n):
    """ Helper function to split_list. """
    nsize = int(len(inputList) / n)

    for i in range(0, len(inputList), nsize):
        yield inputList[i:i + nsize]


def split_list(inputList, n):
    """ Divides inputList into n separate lists. """
    return list(chunks(inputList, n))
