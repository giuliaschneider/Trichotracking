import pandas as pd


def import_df(filepath):
    df = pd.read_csv(filepath)
    return df
