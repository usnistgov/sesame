from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
 
import numpy as np 

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as
FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import logging

from sesame import plotter

class MplWindow(QDialog):
    def __init__(self, parent=None):
        super(MplWindow, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.plot()

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot(self, system=None, data=None):
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)
        if system and isinstance(data, np.ndarray):
            p = plotter.plot(system, data, show=False)

        elif data == 'line':
            p = plotter.plot_line_defects(system, show=False)

        elif data == 'plane':
            p = plotter.plot_plane_defects(system, show=False)

        self.canvas.draw()

