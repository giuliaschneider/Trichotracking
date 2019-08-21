import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import ListedColormap

from trichotracking.iofiles import loadImage
from postprocess import getDarkPhases



def get_mycmap():
    t10 = cm.get_cmap('tab10',3)
    mycmap = np.zeros((4,4))
    mycmap[2:] = t10(np.arange(2))
    mycmap[1] = t10(3)
    #mycmap[:,3] *= 0.8
    new_cmp = ListedColormap(mycmap)
    return new_cmp


class OverlapAnimation(animation.TimedAnimation):
    def __init__(self, list_img, list_bw, df_tracks, darkphases):

        self.list_img = list_img
        self.list_bw = list_bw
        self.times = mpl.dates.date2num(df_tracks.time.values)
        self.length1 = df_tracks.length1.values
        self.length2 = df_tracks.length2.values
        self.length_overlap = df_tracks.length_overlap.values
        self.xlov = df_tracks.xlov
        self.ylov = df_tracks.ylov
        self.overlap1 =  self.length_overlap / self.length1
        self.overlap2 =  self.length_overlap / self.length2

        self.darkphases = darkphases
        if darkphases is not None:
            self.dark = getDarkPhases(darkphases, self.times)

        fig = plt.figure()
        self.ax1 = fig.add_subplot(312)
        ax2 = fig.add_subplot(313, sharex=self.ax1)
        ax3 = fig.add_subplot(321)
        ax4 = fig.add_subplot(322)

        plt.setp(self.ax1.get_xticklabels(), visible=False)
        min_y = 1.1*np.min(self.xlov[~np.isnan(self.xlov)])
        max_y = 1.1*max( np.max(self.length1[~np.isnan(self.length1)]),
                         np.max(self.length2[~np.isnan(self.length2)]))
        self.line1, = self.ax1.plot_date([], [], '-')
        self.line2, = self.ax1.plot_date([], [], '-')
        self.line2b, = self.ax1.plot_date([], [], '-')
        self.line2c, = self.ax1.plot_date([], [], '-')
        self.line2d, = self.ax1.plot_date([], [], '-')
        self.ax1.set_ylabel('Length [px]')
        self.ax1.set_xlim(np.min(self.times), np.max(self.times))
        self.ax1.set_ylim(min_y, max_y)
        self.title = self.ax1.set_title('')

        self.line3, = ax2.plot_date([], [], '-')
        self.line4, = ax2.plot_date([], [], '-')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Overlap')
        ax2.set_xlim(np.min(self.times), np.max(self.times))
        ax2.set_ylim(0, 1)
        ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))

        if self.darkphases is not None:
            import matplotlib.transforms as mtransforms
            trans = mtransforms.blended_transform_factory(self.ax1.transData,
                                                          self.ax1.transAxes)
            self.ax1.fill_between(self.times, 0, 1, where=self.dark,
                transform=trans, facecolor='black', alpha=0.3)
            trans = mtransforms.blended_transform_factory(ax2.transData,
                                                          ax2.transAxes)
            ax2.fill_between(self.times, 0, 1, where=self.dark,
                transform=trans, facecolor='black', alpha=0.3)


        img = loadImage(self.list_img[0])[0]
        self.img1 = ax3.imshow(img, cmap='gray', animated=True)
        ax3.set_axis_off()

        bw = loadImage(self.list_bw[0])[0]
        mycmap = get_mycmap()
        self.img2 = ax3.imshow(bw, alpha=0.8, cmap=mycmap, animated=True)
        ax4.set_axis_off()

        animation.TimedAnimation.__init__(self, fig, interval=1000, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        print(i)

        self.ax1.set_title("Frame = {}".format(i))

        img = loadImage(self.list_img[i])[0]
        self.img1.set_array(img)

        bw = loadImage(self.list_bw[i])[0]
        self.img2.set_array(bw)

        self.line1.set_data(self.times[:i], self.length1[:i])
        self.line2.set_data(self.times[:i], self.length2[:i])
        self.line2b.set_data(self.times[:i], self.length_overlap[:i])
        self.line2c.set_data(self.times[:i], self.xlov[:i])
        self.line2d.set_data(self.times[:i], self.ylov[:i])

        self.line3.set_data(self.times[:i], self.overlap1[:i])
        self.line4.set_data(self.times[:i], self.overlap2[:i])

        self._drawn_artists = [self.line1, self.line2, self.line2b,
                               self.line2c, self.line2d, self.line3,
                               self.line4, self.img1, self.img2]

    def new_frame_seq(self):
        return iter(range(len(self.list_img)))

    def _init_draw(self):
        lines = [self.line1, self.line2]
        for l in lines:
            l.set_data([], [])
