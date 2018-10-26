# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import numpy as np

from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as
FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

from .common import parseLocation
from .. import plotter


class MplWindow(QWidget):
    def __init__(self):
        super(MplWindow, self).__init__()

        # a figure instance to plot on
        self.figure = Figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # create an axis to have an empty graph
        self.ax = self.figure.add_subplot(111)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # plotting colors for materials
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',\
                       '#bcbd22', '#17becf']
        self.iterColors = iter(self.colors)

    def plotSystem(self, sys, materials, defects):
        # Upon saving a material or a defect, we clear the figure and plot all
        # materials, line defects and the grid

        # We tweak the color map so that no color is displayed when no material
        # has been created yet in parts of the grid
        cmap = get_cmap('Set2')
        cmap = cmap(np.arange(cmap.N)) # get the colors
        cmap[0,-1] = 0 # set alpha of the first color to zero
        cmap = ListedColormap(cmap)

        try:
            self.ax.clear()
            self.toolbar.update()
            # find all relevant coordinates
            nx, ny = sys.nx, sys.ny
            xpts, ypts = sys.xpts, sys.ypts
            if ny == 1:
                ypts = np.array([0, 1])
                ny = 2
            x, y = [], []
            for mat in materials:
                location = parseLocation(mat['location'], sys.dimension)
                if sys.dimension==2:
                    indices = np.array([[i,j] for j in range(ny) for i in range(nx)\
                                    if location((xpts[i], ypts[j]))])
                else:
                    indices = np.array([[i, j] for j in range(ny) for i in range(nx) \
                                        if location((xpts[i]))])
                x.append(indices[:,0])
                y.append(indices[:,1])

                if sys.dimension == 1:
                    self.ax.plot(xpts[x[-1]], np.ones_like(xpts[x[-1]]) / 2., lw=50)
                    #self.ax.plot(xpts[x], np.ones_like(xpts[x])/2., lw=50)
                    self.ax.margins(0)
            # create an array of fake data to be plotted
            d = np.zeros((nx, ny)) - 1
            for idx, (posx, posy) in enumerate(zip(x, y)):
                d[posx, posy] = idx + 1

            if sys.ny > 1:
                if (d > 0).all():
                    cmap = 'Set2'
                self.ax.pcolormesh(xpts, ypts, d.T, cmap=cmap)
                plotter.plot_line_defects(sys, fig=self.figure)

            # plot grid on top
            if sys.ny == 1:
                self.ax.plot([xpts[0],xpts[0]],[0,1],'k',linewidth=.25)
                for xpt in xpts[1:]:
                    self.ax.plot([xpt,xpt],[0.4,0.6],'k',linewidth=.25)
                
            if sys.ny > 1:
                for xpt in xpts:
                    self.ax.plot([xpt,xpt],[ypts[0],ypts[-1]],'k',linewidth=.25)
                for ypt in ypts:
                    self.ax.plot([xpts[0],xpts[-1]],[ypt,ypt],'k',linewidth=.25)
            
            if sys.ny == 1:
                self.ax.get_yaxis().set_visible(False)
            else:
                self.ax.get_yaxis().set_visible(True)

            self.ax.set_xlabel('x [cm]')
            if sys.ny > 1:
                self.ax.set_ylabel('y [cm]')

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception:
            pass
