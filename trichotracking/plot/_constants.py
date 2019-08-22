import matplotlib.cm as cm
import matplotlib.pyplot as plt

FIGSIZE = (8.25 / 2.54, 5.75 / 2.54)
FIGSIZE_BARS = (12 / 2.54, 6 / 2.54)
FIGSIZE_TS = (17.5 / 2.54, 2.5 / 2.54)
MARKERSIZE = 2.5
FONTSIZE = 11

AGG_COLOR = cm.tab10(0)
NAGG_COLOR = cm.tab10(1)
AGG_LABEL = 'Aggregating'
NAGG_LABEL = 'Non-aggregating'

REV_COLOR = cm.tab10(6)
NREV_COLOR = cm.tab10(7)

NSPLIT_COLOR = cm.Set2(0)
SPLIT_COLOR = cm.Set2(2)

REV_NSPLIT_COLOR = REV_COLOR
REV_SPLIT_COLOR = cm.tab20(13)
NREV_NSPLIT_COLOR = NREV_COLOR
NREV_SPLIT_COLOR = cm.tab20(15)
REV_SPLIT_COLORS = [REV_NSPLIT_COLOR, REV_SPLIT_COLOR, NREV_NSPLIT_COLOR,
                    NREV_SPLIT_COLOR]
REV_SPLIT_COLORS_W = [REV_NSPLIT_COLOR, REV_SPLIT_COLOR, NREV_SPLIT_COLOR]

LOL_COLOR = cm.tab20b(4)
LOL_MA_COLOR = cm.tab20b(6)
POS_COLOR = cm.tab20b(8)

plt.rcParams.update({'lines.markersize': MARKERSIZE})
plt.rcParams.update({'font.size': FONTSIZE})

REV_SPLIT_LABEL = ['Reversing and non-separating pairs', 'Reversing and separating pairs',
                   'Non-reversing and non-separating pairs', 'Non-reversing and separating pairs']
REV_SPLIT_LABEL_W = ['Reversing and non-separating pairs', 'Reversing and separating pairs',
                     'Non-reversing and separating pairs']

V_UNIT = ' [' + u'\u03bc' + 'm/s]'
