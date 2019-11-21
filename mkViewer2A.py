#!/usr/bin/env python
"""
===================
Image Slices Viewer
===================

This is a modification of mkViewer2.py

Scroll through 2D image slices of a 3D array.
https://matplotlib.org/gallery/event_handling/image_slices_viewer.html



C: 2018.12.01
M: 2019.11.15

ver: 0.02
"""
#from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class IndexTracker(object):
    def __init__(self, X, cmap, aspect=1, title='Image', rot=False):

        if not isinstance(X, np.ndarray):
            print('mkp-Viewer2 --->>> X matrix is not a NUMPY object type!!!')
            return

        self.fig = plt.figure()
        self.fig.subplots_adjust(left=0.2, bottom=0.2)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('use scroll wheel to navigate images')
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0.2, bottom=0.2)

        self.rot = rot
        self.cmap = cmap
        self.X = X
        self.title = title
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.aspect = aspect


        mn, mx = X.min(), X.max()
        av = (mn + mx) / 2
        txt = "min={:.2f}, av={:.2f}, max={:.2f}, dtype={}, shape={}".format(mn, av, mx, X.dtype, X.shape)
        self.fig.text(0.05,0.03, txt)

        self.im = self.ax.imshow(self.X[:, :, self.ind], vmin=mn, vmax=mx)
        self.im.set_cmap(self.cmap)
        self.update()

        slidercolor = 'lightgoldenrodyellow'
        slideraxes = self.fig.add_axes([0.25, 0.1, 0.65, 0.05], facecolor=slidercolor)
        self.slider = Slider(slideraxes, 'Slice.', 0, self.slices-1, valinit=self.ind, valfmt='%i')

        self.slider.on_changed(self.update_slider)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        plt.show()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.slider.set_val(self.ind)
        self.update()

    def update_slider(self, val):
        self.ind = int(val);
        self.update()

    def update(self):
        if self.rot:
            self.im.set_data(self.X[:, :, self.ind].T)
        else:
            self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_title('%s - slice %s' % (self.title, self.ind))
        self.ax.set_aspect(self.aspect)
        self.im.axes.figure.canvas.draw()




if __name__ =='__main__':

    import sys
    import nibabel as nib

    from mkReader import mkReader


    if len(sys.argv) < 2:
        X = np.random.randint(0, 100, size=(100, 100, 100))
        print("\nUsage: filename.[npy/nii/nii.gz] [cmap]\n")
        filename = 'Random image'
        aspect = 1

    else:
        filename = str(sys.argv[1])
        reader = mkReader(filename)
        #reader.setRawParams()
        X = reader.loadData()
        pixdim = reader.getPixDimFromNiiHeader()
        aspect = pixdim[1] / pixdim[0]
        print(aspect)

    # CMAP (jako drugi argument)
    if len(sys.argv) == 3:
        cmap = str(sys.argv[2])
        #print(cmap)
    else:
        cmap = 'gray'


#    filename = '018_S_4313_c3d.nii'
#    reader = mkReader(filename)
#    X = reader.loadData()
    tracker = IndexTracker(X, cmap, aspect, title=filename)

