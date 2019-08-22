import numpy as np

from trichotracking.dfmanip import combineNanCols, groupdf, listToColumns


def getSingleFilamentTracks(df, dfg, aggTracks):
    """ Returns trackNrs of for sure single filaments. """
    # Calculate length variation dependent on length
    dfg.loc[:, 'length_std_n'] = dfg.loc[:, 'length_std'] \
                                 / dfg.loc[:, 'length_mean']

    dfg.loc[:, 'area_std_n'] = dfg.loc[:, 'area_std'] / dfg.loc[:, 'area_mean']

    fsingleTracks1 = dfg[((~dfg['trackNr'].isin(aggTracks))
                          & (dfg['eccentricity_mean'] > 0.98)
                          & (dfg['length_std_n'] < 0.04)
                          & (dfg['length_count'] > 4))].trackNr.values

    fsingleTracks2 = dfg[((~dfg['trackNr'].isin(aggTracks))
                          & (dfg['eccentricity_mean'] > 0.93)
                          & (dfg['length_std_n'] < 0.02)
                          & (dfg['length_count'] > 4))].trackNr.values

    fsingleTracks = np.append(fsingleTracks1, fsingleTracks2)
    fsingleTracks = np.unique(fsingleTracks)

    return fsingleTracks


def get2FilamentTracks(df, dfg, dfagg, fsingleTracks=[]):
    """ Returns trackNrs of for aligned 2 filament particles. """
    ffilTracks = dfagg[dfagg.n == 2].trackNr.values

    filAlignedTracks = dfg[((dfg['trackNr'].isin(ffilTracks))
                            & (dfg['ews_mean'] < 0.02)
                            & (dfg['length_count'] > 4))].trackNr.values
    filCrossTracks = dfg[((dfg['trackNr'].isin(ffilTracks))
                          & (dfg['ews_mean'] >= 0.02)
                          & (dfg['length_count'] > 4))].trackNr.values
    if len(fsingleTracks) > 0:
        ffilTracks2 = dfagg[((dfagg.trackNr.isin(ffilTracks))
                              & ((dfagg.tracks00.isin(fsingleTracks)) | (dfagg.tracks00.isnull()))
                              & ((dfagg.tracks01.isin(fsingleTracks)) | (dfagg.tracks01.isnull()))
                              & ((dfagg.tracks10.isin(fsingleTracks)) | (dfagg.tracks10.isnull()))
                              & ((dfagg.tracks11.isin(fsingleTracks)) | (dfagg.tracks11.isnull())))].trackNr.values
        filAlignedTracks = np.intersect1d(filAlignedTracks, ffilTracks2)
        print(filAlignedTracks)
        filCrossTracks = np.intersect1d(filCrossTracks, ffilTracks2)
    return filAlignedTracks, filCrossTracks


def segment_filaments(df, dfagg):
    """ Segments particle tracks into single and fil-fil tracks.

    Postional arguments:
    --------------------
    dfaggFile :  path to dfagg-textfile
    dflinkedFile :   path to dflinked - textfile

    Returns:
    --------
    singleTracks :  trackNrs of certain single tracks
    ffilTracks :  trackNrs of aligned filament-filaments iteractions
    dfagg :  dataframe of aggegrates
    df :  dataframe of all tracks
    dfg : grouped dataframe
    """

    # Split list into single columns
    dfagg = listToColumns(dfagg, 'tracks0', ['tracks00', 'tracks01'])
    dfagg = listToColumns(dfagg, 'tracks1', ['tracks10', 'tracks11'])
    # Create column which has either trackNr before merge or after split
    dfagg = combineNanCols(dfagg, 'deftracks', 'tracks0', 'tracks1')
    # Split column
    dfagg = listToColumns(dfagg, 'deftracks', ['deftrack1', 'deftrack2'])
    aggTracks = dfagg.trackNr.values

    # Import dflinked
    if not 'ews' in df.keys():
        df['ews'] = df.ew2 / df.ew1
    # Group
    dfg = groupdf(df)

    fsingleTracks = getSingleFilamentTracks(df, dfg, aggTracks)
    filAlignedTracks, filCrossTracks = get2FilamentTracks(df, dfg, dfagg, fsingleTracks)

    return fsingleTracks, filAlignedTracks, filCrossTracks
