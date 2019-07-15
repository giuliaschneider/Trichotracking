

def meanOfList(inputList):
    """ Calculates the mean value of inputList."""
    if not inputList:
        return None
    else:
        return float(sum(inputList)) / len(inputList)
