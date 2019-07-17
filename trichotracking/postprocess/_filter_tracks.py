import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace

def attachTracks(df, track1, track2):
    """ Attaches track2 to track1. """
    track1_avg_length1 = df.loc[df.track==track1, 'length1'].mean()
    track2_avg_length1 = df.loc[df.track==track2, 'length1'].mean()
    track2_avg_length2 = df.loc[df.track==track2, 'length2'].mean()

    if (np.abs(track2_avg_length1-track1_avg_length1) >
        np.abs(track2_avg_length2-track1_avg_length1)):
        df.loc[df.track==track2, ['length1', 'length2']] \
            = df.loc[df.track==track2, ['length2', 'length1']].values
        df.loc[df.track==track2, ['overlap1', 'overlap2']] \
            = df.loc[df.track==track2, ['overlap2', 'overlap1']].values
    df.loc[df.track==track2, 'track'] = track1
    return df





def cleanTracks(df):
    """ Removes all datapoints above/below 15% to np.nan."""
    labels = np.unique(df.label)
    p = 0.15
    cols = ["length1", "length2", "length_overlap", "xlov", "pos_short"]
    for l in labels:
        ml1 = df[df.label==l].length1.mean()
        df = df[~((df.label==l) &
               ((df.length1>(1+p)*ml1) | (df.length1<(1-p)*ml1)))]
        ml2 = df[df.label==l].length2.mean()
        df = df[~((df.label==l) &
               ((df.length2>(1+p)*ml2) | (df.length2<(1-p)*ml2)))]
        """df.loc[((df.label==l) &
               ((df.length2>(1+p)*ml2) | (df.length2<(1-p)*ml2))),
                cols] = np.nan"""
    return df


def filterTracks(df, minDistThresh, maxDistThresh, minFrames=15,
                 linkFrames=10, intThresh=None):
    """ Returns dataframe containing mostly fil-fil interactions. """
    df = df.sort_values(by=["trackNr", "frame"])
    df['ind'] = np.arange(df.shape[0])

    # Get all track and frames with no overlap (nov)
    if intThresh is None:
        intThresh = 255
        df["nov"] = (((df.dist<minDistThresh) | ( df.dist > maxDistThresh)))
    else:
        df["nov"] = ( (df.minI > intThresh)
                    |((df.dist<minDistThresh) | ( df.dist > maxDistThresh)))

    # Number blocks of overlap / no overlap1
    df["grp_nov"] = (df.nov != df.nov.shift()).cumsum()
    dfg = df.groupby(["trackNr", "grp_nov"], as_index=False)
    grouped = dfg.mean()[['trackNr', 'grp_nov', 'nov']]
    grouped['nFrames']  = dfg.count().nov

    # Link blocks with no overlap < linkFrames, if surrounded by overlap
    novUpdate = grouped.loc[((grouped.nov)
                   &(grouped.trackNr == grouped.trackNr.shift())
                   &(grouped.trackNr == grouped.trackNr.shift(periods=-1))
                   &(grouped.nov.shift() == 0)
                   &(grouped.nov.shift(periods=-1) == 0)
                   &(grouped.nFrames<=linkFrames)
                   &(grouped.nFrames.shift()>1)
                   &(grouped.nFrames.shift(periods=-1)>1)
                   ), ['trackNr', 'grp_nov']]
    for track, grp in zip(novUpdate.trackNr, novUpdate.grp_nov):
        df.loc[(df.trackNr == track)&(df.grp_nov == grp), 'nov'] = False

    # Overlap with < minFrames are not considered
    df["grp_nov"] = (df.nov != df.nov.shift()).cumsum()
    dfg = df.groupby(["trackNr", "grp_nov"], as_index=False)
    grouped = dfg.mean()[['trackNr', 'grp_nov', 'nov']]
    grouped['nFrames']  = dfg.count().nov
    novUpdate = grouped.loc[((~grouped.nov)
                            &(grouped.nFrames<minFrames)),
                            ['trackNr', 'grp_nov']]
    for track, grp in zip(novUpdate.trackNr, novUpdate.grp_nov):
        df.loc[(df.trackNr == track)&(df.grp_nov == grp), 'nov'] = True

    df = df[~df.nov]
    return df


def getStartEndTimes(df, minDistTh):
    """ Returns the start, end times of a fil-fil interaction"""

    startTimes = []
    endTimes = []
    labels = np.unique(df.trackNr)
    for label in labels:
        frames = df[(df.trackNr==label) & (df.dist>minDistTh) ].frame
        try:
            startTimes.append(frames.iloc[0])
        except:
            set_trace()
        endTimes.append(frames.iloc[-1])
    return startTimes, endTimes


def trimTrack(df, tracks, t0s, tends=None):
    """ Cuts track frames outside t0-tend.

    Keyword arguments:
    df --  dataframe
    track -- trackNr to be trimmed
    t0 -- starting frame nr
    tend -- ending frame nr
    """

    if type(tracks) is int or type(tracks) is float:
        tracks = [tracks]
        t0s = [t0s]
        tends = [tends]

    if type(tracks) is np.ndarray:
        tracks = tracks.tolist()
    if type(t0s) is np.ndarray:
        t0s = t0s.tolist()
    if type(tends) is np.ndarray:
        tends = tends.tolist()

    for track, t0, tend in zip(tracks, t0s, tends):
        if t0 is not None:
            df = df[((df.trackNr!=track) | ((df.trackNr==track) & (df.frame>=t0)))]

        if tend is not None:
            df = df[((df.trackNr!=track) | ((df.trackNr==track) & (df.frame<=tend)))]

    return df
