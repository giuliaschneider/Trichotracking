import os.path
import numpy as np
import numpy.linalg as la
import numpy.fft as fft
import pandas as pd
import matplotlib as mpl
import scipy.signal
from datetime import datetime

from IPython.core.debugger import set_trace



def calcT(df):
    """ Calculates the T Statistic."""
    tp = df.loc[df.pos_diff > 0, "pos_diff"].abs().sum()
    tm = df.loc[df.pos_diff < 0, "pos_diff"].abs().sum()
    if (tp+tm) == 0:
        return np.nan
    else:
        return (tp - tm) / (tp + tm)
