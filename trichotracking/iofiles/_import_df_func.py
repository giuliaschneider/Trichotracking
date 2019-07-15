import numpy as np

from IPython.core.debugger import set_trace

def extractPixelListFromString(str_list):
    if isinstance(str_list, str):
        test = np.fromstring(str_list[1:-1], sep=', ').astype(int)
    else:
        test = np.array(str_list)
    return test


def extractValuesFromListOfString(list_str):
    """ Returns a list of list of values. """
    values_list = []
    for x in list_str:
        values_list.append(extractPixelListFromString(x).tolist())
    return values_list
