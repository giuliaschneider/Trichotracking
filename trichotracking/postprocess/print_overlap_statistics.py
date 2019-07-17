

def getMeanLOL(dfg, file):
    file.write("{:d};".format(dfg.xlov_ma_abs.count()))
    file.write("{:.2f};".format(dfg.xlov_ma_abs.mean()))
    file.write("{:.2f};".format(dfg.xlov_ma_abs.std()))
    file.write('\n')

def printStatistics(group):
    print("Nr Tracks = {}".format(group.ngroups))
    trackLengths = group.length1.count()
    print(trackLengths)
    print("Avg track length = {}".format(trackLengths.mean()))
    print("Median track length = {}".format(trackLengths.median()))
    print("Std track length = {}".format(trackLengths.std()))


def printTStatistics(df):
    labels = np.unique(df.label)
    for label in labels:
        print("Label = {}".format(label))
        # All
        #set_trace()
        T = calcT(df[(df.label==label)])
        print("All data, T = {} ".format(T))
        # LOO > 0
        T = calcT(df[(df.label==label) & (df.xlov_ma>0)])
        print("LOO > 0, T = {} ".format(T))
        # LOO < 0
        T = calcT(df[(df.label==label) & (df.xlov_ma<0)])
        print("LOO < 0, T = {} ".format(T))
    print("All Tracks")
    T = calcT(df)
    print("All data, T = {} ".format(T))
    # LOO > 0
    T = calcT(df[(df.xlov_ma>0)])
    print("LOO > 0, T = {} ".format(T))
    # LOO < 0
    T = calcT(df[(df.xlov_ma<0)])
    print("LOO < 0, T = {} ".format(T))



def printMeanXLOL(df, saveDir, title):
    filename = os.path.join(saveDir, "meanLol.txt")
    file = open(filename, 'a')
    file.write(title)
    file.write('\n')
    file.write("Data;n;Mean;Std \n")

    # All Data
    file.write("All Data;")
    df_peaks = df[(df.peaks==1) & (df.xlov_ma.abs()>0.01)]
    dfg = df_peaks.groupby('label').mean()
    getMeanLOL(dfg, file)

    # Dark Data
    file.write("Dark;")
    df_peaks = df[(df.peaks==1) & (df.dark==1) & (df.xlov_ma.abs()>0.01)]
    dfg = df_peaks.groupby('label').mean()
    getMeanLOL(dfg, file)

    # Dark Data
    file.write("Light;")
    df_peaks = df[(df.peaks==1) & (df.dark==0) & (df.xlov_ma.abs()>0.01)]
    dfg = df_peaks.groupby('label').mean()
    getMeanLOL(dfg, file)
