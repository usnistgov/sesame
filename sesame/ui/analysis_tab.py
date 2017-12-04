# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
 
from ast import literal_eval as ev
import numpy as np 
import logging

from .plotbox import *
from .common import parseSettings, slotError
from ..analyzer import Analyzer
from ..plotter import plot


class Analysis(QWidget):
    def __init__(self, parent):
        super(Analysis, self).__init__(parent)

        self.table = parent

        self.tabLayout = QVBoxLayout()
        self.setLayout(self.tabLayout)

        self.hlayout = QHBoxLayout()
        self.tabLayout.addLayout(self.hlayout)


        #==============================================
        # Upload data and settings
        #==============================================
        prepare = QVBoxLayout()
        self.hlayout.addLayout(prepare)

        FileBox = QGroupBox("Import data")
        dataLayout = QVBoxLayout()
        self.dataBtn = QPushButton("Select files...")
        self.dataBtn.clicked.connect(self.browse)
        self.dataList = QListWidget()
        self.dataList.setSelectionMode(QAbstractItemView.ExtendedSelection)
        dataLayout.addWidget(self.dataBtn)
        dataLayout.addWidget(self.dataList)
        FileBox.setLayout(dataLayout)
        prepare.addWidget(FileBox)

        twoDBox = QGroupBox("Surface plot")
        twoDLayout = QVBoxLayout()
        self.quantity = QComboBox()
        quantities = ["Choose one", "Electron quasi-Fermi level",\
        "Hole quasi-Fermi level", "Electrostatic potential",\
        "Electron density", "Hole density", "Bulk SRH recombination",\
        "Electron current", "Hole current"]
        self.quantity.addItems(quantities)
        twoDLayout.addWidget(self.quantity)
        self.plotBtnS = QPushButton("Plot")
        self.plotBtnS.clicked.connect(self.surfacePlot)
        twoDLayout.addWidget(self.plotBtnS)
        twoDBox.setLayout(twoDLayout)
        prepare.addWidget(twoDBox)

        oneDBox = QGroupBox("Linear plot")
        oneDLayout = QVBoxLayout()
        form = QFormLayout()
        self.Xdata = QLineEdit()
        form.addRow("X data", self.Xdata)
        self.quantity2 = QComboBox()
        quantities = ["Choose one", "Band diagram",\
        "Electron quasi-Fermi level", "Hole quasi-Fermi level",\
        "Electrostatic potential","Electron density",\
        "Hole density", "Bulk SRH recombination",\
        "Electron current along x", "Electron current along y",\
        "Hole current along x", "Hole current along y",\
        "Full steady state current"]
        self.quantity2.addItems(quantities)
        form.addRow("Y data", self.quantity2)
        oneDLayout.addLayout(form)
        self.plotBtn = QPushButton("Plot")
        self.plotBtn.clicked.connect(self.linearPlot)
        oneDLayout.addWidget(self.plotBtn)
        oneDBox.setLayout(oneDLayout)
        prepare.addWidget(oneDBox)
 

        #==============================================
        # Surface plot
        #==============================================
        self.surfaceLayout = QVBoxLayout()
        self.hlayout.addLayout(self.surfaceLayout)

        self.surfaceBox = QGroupBox("Surface plot")
        self.vlayout = QVBoxLayout()
        self.surfaceBox.setLayout(self.vlayout)
        self.surfaceLayout.addWidget(self.surfaceBox)

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

        self.linearFig = MplWindow()
        self.vlayout2.addWidget(self.linearFig)

    def browse(self):
        dialog = QFileDialog()
        paths = dialog.getOpenFileNames(self, "Select files")[0]
        for i, path in enumerate(paths):
            self.dataList.insertItem (i, path )

    @slotError("bool")
    def surfacePlot(self, checked):
        # get system
        settings = self.table.build.getSystemSettings()
        system = parseSettings(settings)

        # get data from file
        files = [x.text() for x in self.dataList.selectedItems()]
        if len(files) == 0:
            return
        elif len(files) > 1:
            msg = QMessageBox()
            msg.setWindowTitle("Processing error")
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Select a single data file for a surface plot.")
            msg.setEscapeButton(QMessageBox.Ok)
            msg.exec_()
            return
        else:
            fileName = files[0]
            data = np.load(fileName)

            # make an instance of the Analyzer
            az = Analyzer(system, data)

            # scalings
            vt = system.scaling.energy
            N  = system.scaling.density
            G  = system.scaling.generation
            j  = system.scaling.current

            # plot
            txt = self.quantity.currentText()
            self.surfaceFig.figure.clear()
            dmap = None
            if txt == "Electron quasi-Fermi level":
                dmap = vt * az.efn
            if txt == "Hole quasi-Fermi level":
                dmap = vt * az.efp
            if txt == "Electrostatic potential":
                dmap = vt * az.v
            if txt == "Electron density":
                dmap = N * az.electron_density()
            if txt == "Hole density":
                dmap = N * az.hole_density()
            if txt == "Shockley-Read-Hall recombination":
                dmap = G * az.bulk_srh_rr()
            
            if dmap != None:
                plot(system, dmap, scale=1e-6, cmap='viridis',\
                     fig=self.surfaceFig.figure)
 
            if txt == "Electron current":
                az.current_map(True, 'viridis', 1e6, fig=self.surfaceFig.figure)

            if txt == "Hole current":
                az.current_map(False, 'viridis', 1e6, fig=self.surfaceFig.figure)
            self.surfaceFig.canvas.draw()

    @slotError("bool")
    def linearPlot(self, checked):
        # get data files names
        files = [x.text() for x in self.dataList.selectedItems()]
        if len(files) == 0:
            return

        # get system
        settings = self.table.build.getSystemSettings()
        system = parseSettings(settings)

        # scalings
        vt = system.scaling.energy
        N  = system.scaling.density
        G  = system.scaling.generation
        j  = system.scaling.current

        # clear the figure
        self.linearFig.figure.clear()

        # test what kind of plot we are making

        Xdata = ev(self.Xdata.text())
        Ytxt = self.quantity2.currentText()

        # loop over the files and plot
        for fileName in files:
            data = np.load(fileName)
            az = Analyzer(system, data)

            # get sites and coordinates of a line or else
            if isinstance(Xdata[0], tuple):
                X, sites = az.line(system, Xdata[0], Xdata[1])
                X = X * system.scaling.length * 1e6 # set length in um
                p1, p2 = Xdata
            else:
                X = Xdata

            # get the corresponding Y data
            if txt == "Electron quasi-Fermi level":
                Ydata = vt * az.efn[sites]
            if txt == "Hole quasi-Fermi level":
                Ydata = vt * az.efp[sites]
            if txt == "Electrostatic potential":
                Ydata = vt * az.v[sites]
            if txt == "Electron density":
                Ydata = N * az.electron_density(location=(p1, p2))
            if txt == "Hole density":
                Ydata = N * az.hole_density(location=(p1, p2))
            if txt == "Shockley-Read-Hall recombination":
                Ydata = G * az.bulk_srh_rr(location=(p1, p2))
            if txt == "Electron current along x":
                Ydata = J * az.electron_current(component='x', location=(p1, p2))
            if txt == "Hole current along x":
                Ydata = J * az.hole_current(component='x', location=(p1, p2))
            if txt == "Electron current along y":
                Ydata = J * az.electron_current(component='y', location=(p1, p2))
            if txt == "Hole current along x":
                Ydata = J * az.hole_current(component='y', location=(p1, p2))
            if txt == "Full steady state current":
                Ydata = J * az.full_current()

            # plot
            if txt != "Band diagram": # everything except band diagram
                ax = self.linearFig.figure.add_subplot(111)
                ax.plot(X, Ydata)
            else:
                az.band_diagram((Xdata[0], Xdata[1]), fig=self.linearFig.figure)
                
            self.linearFig.canvas.draw()
