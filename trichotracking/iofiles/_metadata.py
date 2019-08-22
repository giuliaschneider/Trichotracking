import os.path
from datetime import datetime

__all__ = ['getTime', 'getTime_timestamp']


def getTime(filePath):
    """ Returns the time of last modification of filePath in s.  """
    return os.path.getmtime(filePath)


def getTime_timestamp(filePath):
    """Returns the time of last modification of filePath as timestamp"""
    return datetime.fromtimestamp(getTime(filePath))
