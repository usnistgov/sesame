from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
 
import numpy as np 
import logging

from .plotbox import *
from .makeSystem import parseSettings
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
        quantities = ["Choose one", "Shockley-Read-Hall Recombination",\
                      "Electron current", "Hole current"]
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
        "Hole density", "Shockley-Read-Hall Recombination",\
        "Electron current along x", "Electron current along y",\
        "Hole current along x", "Hole current along y",\
        "Full current along x", "Full current along y",\
        "Norm of the current"]

        self.quantity2.addItems(quantities)
        self.quantity2.currentIndexChanged.connect(self.linearPlot)
        h.addWidget(self.quantity2)
        h.addStretch()
        self.vlayout2.addLayout(h)

        self.linearFig = MplWindow()
        self.vlayout2.addWidget(self.linearFig)

        ls2 = QHBoxLayout()
        ls2.addWidget(QLabel("First point"))
        ls2.addWidget(QLineEdit("(x1, y1)"))
        ls2.addWidget(QLabel("Second point"))
        ls2.addWidget(QLineEdit("(x2, y2)"))
        self.vlayout2.addLayout(ls2)


    def surfacePlot(self):
        # get data from file
        fileName = self.fileName.text()

        # get system
        settings = self.table.settingsBox.get_settings()
        system = parseSettings(settings)

        # make an instance of the Analyzer
        az = Analyzer(system, fileName)

        # plot
        txt = self.quantity.currentText()
        self.surfaceFig.clear()
        if txt == "Electron current":
            p = az.current_map(True, 'viridis', 1e6, show=False)
        if txt == "Hole current":
            p = az.current_map(False, 'viridis', 1e6, show=False)

        self.surfaceFig.canvas.draw()

    def linearPlot(self):
        return
