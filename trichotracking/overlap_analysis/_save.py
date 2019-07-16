import os.path

def save_dfs_to_file(overlapDir, df, df_tracks, mdfo, dflinked):
    df.to_csv(os.path.join(overlapDir, 'dfoverlap.txt'))
    df_tracks.to_csv(os.path.join(overlapDir, 'df_tracks.txt'))
    mdfo.to_csv(os.path.join(overlapDir, 'mdfo.txt'))
    dflinked.to_csv(os.path.join(overlapDir, 'dflinked.txt'))
