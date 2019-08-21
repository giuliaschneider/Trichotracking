import pandas as pd
from trichotracking.dfmanip import combineNanCols, listToColumns

from ._import_df_func import extractValuesFromListOfString as listValues

from IPython.core.debugger import set_trace



def import_df(filepath):
    df = pd.read_csv(filepath)
    return df
