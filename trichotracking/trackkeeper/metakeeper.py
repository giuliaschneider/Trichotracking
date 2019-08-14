import pandas as pd


class Metakeeper:
    def __init__(self, df):
        self.df = df

    @classmethod
    def fromFile(cls, file):
        cols = pd.read_csv(file, index_col=0, nrows=0).columns
        cols = cols[~cols.str.startswith('Unnamed')]
        df = pd.read_csv(file, usecols=cols, index_col=None, header=0)
        return cls(df)

    def addColumn(self, df_new):
        self.df = self.df.merge(df_new, left_on='trackNr', right_on='trackNr', how='left')

    def getDf(self):
        return self.df

    def save(self, file):
        self.df.to_csv(file)
