import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from iofiles import loadImage
from plot.plot_images import hsv2rgb


class TrackingAnimation(animation.TimedAnimation):
    def __init__(self, list_img, df_tracks, nTracks):

        self.list_img = list_img
        self.df_tracks = df_tracks
        self.maxTrackNr = np.max(df_tracks.trackNr)

        fig = plt.figure(figsize=(10, 10))
        self.ax1 = fig.add_subplot(111)

        self.title = self.ax1.set_title('')

        img, self.h, self.w = loadImage(self.list_img[0])
        self.img1 = self.ax1.imshow(img, animated=True)
        self.ax1.set_xlim(0, self.w)
        self.ax1.set_ylim(0, self.h)

        self.scat = self.ax1.scatter([], [], cmap='rainbow')

        self.texts = [self.ax1.text(i,i,'') for i in range(nTracks)]
        animation.TimedAnimation.__init__(self, fig, interval=350, blit=False)
        #self.ax1.set_axis_off()


    def _draw_frame(self, framedata):
        i = framedata
        print(i)

        # Image
        img, height, width = loadImage(self.list_img[i])
        self.img1.set_array(img)
        self.ax1.set_title("Frame = {}".format(i))

        # Scatter plot
        cx = self.df_tracks[self.df_tracks.frame == i].cx
        cy = self.df_tracks[self.df_tracks.frame == i].cy
        trackNr = self.df_tracks[self.df_tracks.frame == i].trackNr
        rgb = hsv2rgb(trackNr/self.maxTrackNr, 1, 1)
        self.scat.set_offsets(np.c_[cx, cy])
        self.scat.set_color((rgb))

        # Texts
        for nr, t in enumerate(self.texts):
            if nr in trackNr.values:
                x = cx[trackNr==nr].values[0]
                y = cy[trackNr==nr].values[0]
                t.set_position((x, y))
                t.set_text(nr)
            else:
                t.set_text('')

        self._drawn_artists = [self.img1, self.scat, self.texts]

    def new_frame_seq(self):
        return iter(range(len(self.list_img)))

    def _init_draw(self):
        pass
