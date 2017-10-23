from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
 
from ast import literal_eval as ev
import numpy as np 
import logging

from .plotbox import *
from .common import parseSettings, slotError
from ..analyzer import Analyzer


class Analysis(QWidget):
    def __init__(self, parent):
        super(Analysis, self).__init__(parent)

        self.table = parent

        self.tabLayout = QVBoxLayout()
        self.setLayout(self.tabLayout)

        self.fileLayout = QHBoxLayout()
        self.fileLayout.addWidget(QLabel("File name"))
        self.fileName = QLineEdit()
        self.fileLayout.addWidget(self.fileName)
        self.tabLayout.addLayout(self.fileLayout)

        self.hlayout = QHBoxLayout()
        self.tabLayout.addLayout(self.hlayout)

        #==============================================
        # Surface plot
        #==============================================
        self.surfaceLayout = QVBoxLayout()
        self.hlayout.addLayout(self.surfaceLayout)

        self.surfaceBox = QGroupBox("Surface plot")
        self.vlayout = QVBoxLayout()
        self.surfaceBox.setLayout(self.vlayout)
        self.surfaceLayout.addWidget(self.surfaceBox)

        hh = QHBoxLayout()
        hh.addWidget(QLabel("Plotted quantity"))
        self.quantity = QComboBox()
        quantities = ["Choose one", "Electron current", "Hole current"]
        self.quantity.addItems(quantities)
        self.quantity.currentIndexChanged.connect(self.surfacePlot)
        hh.addWidget(self.quantity)
        hh.addStretch()
        self.vlayout.addLayout(hh)

        self.surfaceFig = MplWindow()
        self.vlayout.addWidget(self.surfaceFig)

        #==============================================
        # Linear plot
        #==============================================
        self.linearLayout = QVBoxLayout()
        self.hlayout.addLayout(self.linearLayout)

        self.linearBox = QGroupBox("Linear plot")
        self.vlayout2 = QVBoxLayout()
        self.linearBox.setLayout(self.vlayout2)
        self.linearLayout.addWidget(self.linearBox)

        h = QHBoxLayout()
        h.addWidget(QLabel("Plotted quantity"))
        self.quantity2 = QComboBox()
        quantities = ["Choose one", "Band diagram",\
        "Electron quasi-Fermi level", "Hole quasi-Fermi level",\
        "Electrostatic potential","Electron density",\
        "Hole density", "Bulk Shockley-Read-Hall recombination",\
        "Electron current along x", "Electron current along y",\
        "Hole current along x", "Hole current along y"]

        self.quantity2.addItems(quantities)
        self.quantity2.currentIndexChanged.connect(self.linearPlot)
        h.addWidget(self.quantity2)
        h.addStretch()
        self.vlayout2.addLayout(h)

        self.linearFig = MplWindow()
        self.vlayout2.addWidget(self.linearFig)

        ls2 = QHBoxLayout()
        ls2.addWidget(QLabel("First point"))
        self.p1 = QLineEdit("x1, y1")
        ls2.addWidget(self.p1)
        ls2.addWidget(QLabel("Second point"))
        self.p2 = QLineEdit("x2, y2")
        ls2.addWidget(self.p2)
        self.vlayout2.addLayout(ls2)

    def getAnalyzer(self):
        # get system
        settings = self.table.settingsBox.get_settings()
        system = parseSettings(settings)

        # get data from file
        fileName = self.fileName.text()
        data = np.load(fileName)

        # make an instance of the Analyzer
        az = Analyzer(system, data)
        return az, system

    @slotError("bool")
    def surfacePlot(self, checked):
        az, system = self.getAnalyzer()

        # plot
        txt = self.quantity.currentText()
        self.surfaceFig.figure.clear()
        if txt == "Electron current":
            az.current_map(True, 'viridis', 1e6, fig=self.surfaceFig.figure)

        if txt == "Hole current":
            az.current_map(False, 'viridis', 1e6, fig=self.surfaceFig.figure)
        self.surfaceFig.canvas.draw()

    @slotError("bool")
    def linearPlot(self, checked):
        az, system = self.getAnalyzer()

        # gather input from user
        txt = self.quantity2.currentText()
        p1 = ev(self.p1.text())
        p2 = ev(self.p2.text())

        # get sites and coordinates of a line
        X, sites = az.line(system, p1, p2)
        X = X * system.scaling.length * 1e6 # set length in um

        # scalings
        vt = system.scaling.energy
        N  = system.scaling.density
        G  = system.scaling.generation
        j  = system.scaling.current

        # get the corresponding data
        # if txt == "Band diagram":
        if txt == "Electron quasi-Fermi level":
            data = vt * az.efn[sites]
        if txt == "Hole quasi-Fermi level":
            data = vt * az.efp[sites]
        if txt == "Electrostatic potential":
            data = vt * az.v[sites]
        if txt == "Electron density":
            data = N * az.electron_density(location=(p1, p2))
        if txt == "Hole density":
            data = N * az.hole_density(location=(p1, p2))
        if txt == "Shockley-Read-Hall recombination":
            data = G * az.bulk_srh_rr(location=(p1, p2))
        if txt == "Electron current along x":
            data = J * az.electron_current(component='x', location=(p1, p2))
        if txt == "Hole current along x":
            data = J * az.hole_current(component='x', location=(p1, p2))
        if txt == "Electron current along y":
            data = J * az.electron_current(component='y', location=(p1, p2))
        if txt == "Hole current along x":
            data = J * az.hole_current(component='y', location=(p1, p2))

        # plot
        self.linearFig.figure.clear()
        ax = self.linearFig.figure.add_subplot(111)
        ax.plot(X, data)
        self.linearFig.canvas.draw()
