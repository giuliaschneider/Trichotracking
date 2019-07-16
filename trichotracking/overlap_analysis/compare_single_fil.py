from os.path import join
import numpy as np
import pandas as pd
from overlap.analysis_overlap import calcLabel
from overlap.plot_overlap_hist import plotHistNQuantities
from overlap.plot_overlap_ts import *

from IPython.core.debugger import set_trace


def matchLengths(filLengths, singleLength):
    """ Returns the minimal difference between lengths.

    Keyword arguments:
    avg --      list of average past length: (length1, length2)
    current --   list of current length: (length1, length2)

    Returns:
    index of minimal error
    minimal difference
    """
    diff = np.abs(filLengths - singleLength)
    ind = np.argmin(diff)
    indMaxFil = np.argmax(filLengths)

    if ind == indMaxFil:
        label = 1
    else:
        label = 2

    return label


def compare_single_fil(dfMeta, df, df_s, filLabel, singleTracks,
                       saveDir):
    df = df[df.label==filLabel]
    exp = df.exp.values[0]
    chamber = int(filLabel[2])
    labels = calcLabel(singleTracks, chamber, exp[:2])
    df_s = df_s[(df_s.label.isin(labels))].copy()
    df_s['fil'] = np.nan
    dfg = df.groupby('label').median()
    df_s.trackNr = df_s.trackNr.astype('int')
    dfg_single = df_s.groupby('label').median()

    title_base = dfMeta.loc[dfMeta.exp==exp, 'title'].values[0]
    filename_base = join(saveDir, exp, filLabel)
    ylabel = 'Frequency'
    nbins = 50

    filLengths = dfg.loc[:, ['l1_ma', 'l2_ma']].values[0]
    dl = filLengths[1] / filLengths[0]
    legT = r'$l_s/l_l$'+' = {:.2f}'.format(dl)

    if filLengths[0] - filLengths[1] > 80:
        for label in labels:
            singleLength = dfg_single.at[label, 'length_ma']
            fil = matchLengths(filLengths, singleLength)
            df_s.loc[df_s.label==label, 'fil'] = fil

        titles = [title_base + ', ' + filLabel + r' $v_1$',
                  title_base + ', ' + filLabel + r' $v_2$',
                  title_base + ', ' + filLabel + r' $a_1$',
                  title_base + ', ' + filLabel + r' $a_2$']
        filenames = [filename_base  + '_v1',
                     filename_base  + '_v2',
                     filename_base  + '_a1',
                     filename_base  + '_a2']
        xlabels = [r'$v_{filament}$',
                   r'$v_{filament}$',
                   r'$a_{filament}$',
                   r'$a_{filament}$']

        xlims = [(0,2.3),
                 (0,2.3),
                 (0,0.05),
                 (0,0.05)]

        label1 = 'Single Dark'
        label2 = 'Single Light'
        label3 = 'Filament Dark'
        label4 = 'Filament Light'
        labels = [(label1, label2, label3, label4),
                  (label1, label2, label3, label4),
                  ('Single', 'Dark'),
                  ('Single', 'Dark')]

        listOfSs = [(df_s[((df_s.fil==1)&(df_s.dark==1))].v.abs(),
                     df_s[((df_s.fil==1)&(df_s.dark==0))].v.abs(),
                     df[df.dark==1].v1_abs,
                     df[df.dark==0].v1_abs),
                    (df_s[((df_s.fil==2)&(df_s.dark==1))].v.abs(),
                     df_s[((df_s.fil==2)&(df_s.dark==0))].v.abs(),
                     df[df.dark==1].v2_abs,
                     df[df.dark==0].v2_abs),
                    (df_s[df_s.fil == 1].a.abs(),
                     df.a1.abs()),
                    (df_s[df_s.fil == 2].a.abs(),
                     df.a2.abs())]

        legTs = [legT+r', $l$'+' = {:.2f}µm'.format(filLengths[0]),
                 legT+r', $l$'+' = {:.2f}µm'.format(filLengths[1]),
                 legT+r', $l$'+' = {:.2f}µm'.format(filLengths[0]),
                 legT+r', $l$'+' = {:.2f}µm'.format(filLengths[1])]

        for listOfS, label, title, filename, xlabel, xlim, legT in \
         zip(listOfSs, labels, titles, filenames, xlabels, xlims, legTs):


            plotHistNQuantities(listOfS, label, title, xlabel, xlim,
                                nbins, filename, legT)


        df_s['v1_ma'] = df_s[df_s.fil==1].v_ma
        df_s['v2_ma'] = df_s[df_s.fil==2].v_ma
        df_s['a1'] = df_s[df_s.fil==1].a
        df_s['a2'] = df_s[df_s.fil==2].a
        df1 = df[['exp','label', 'time', 'peaks', 'v1_ma', 'v2_ma',
                  'l1_ma', 'l2_ma', 'v_pos_ma', 'a1', 'a2']]
        df2 = df_s[['label', 'time', 'v1_ma', 'v2_ma', 'a1', 'a2']]
        df_tot = pd.concat([df1, df2], sort=False)
        df_tot.sort_values(by=['label', 'time'], inplace=True)
        plotTracksV(df_tot, dfMeta, saveDir)


    else:
        # Both filaments
        s1 = df_s.v.abs()
        label1 = "Single"
        s2 =  pd.concat([df.v1_abs, df.v2_abs])
        label2 = "Filament-Filament"
        title = title_base + ' Both filaments'
        filename = filename_base  + '_vsingle'
        plotHistNQuantities(s1, s2, label1, label2, title,
                                  xlabel, xlims, nbins, filename)
