# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
 
import os
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
        width = 300
        self.hlayout.addLayout(prepare)

        FileBox = QGroupBox("Import data")
        FileBox.setMaximumWidth(width)
        dataLayout = QVBoxLayout()

        # Select and remove buttons
        btnsLayout = QHBoxLayout()
        self.dataBtn = QPushButton("Upload files...")
        self.dataBtn.clicked.connect(self.browse)
        self.dataRemove = QPushButton("Remove selected")
        self.dataRemove.clicked.connect(self.remove)
        btnsLayout.addWidget(self.dataBtn)
        btnsLayout.addWidget(self.dataRemove)
        dataLayout.addLayout(btnsLayout)

        # List itself
        self.filesList = []
        self.dataList = QListWidget()
        self.dataList.setSelectionMode(QAbstractItemView.ExtendedSelection)
        dataLayout.addWidget(self.dataList)
        FileBox.setLayout(dataLayout)
        prepare.addWidget(FileBox)

        # Surface plot
        twoDBox = QGroupBox("Surface plot")
        twoDBox.setMaximumWidth(width)
        twoDLayout = QVBoxLayout()
        self.quantity = QComboBox()
        quantities = ["Choose one", "Electron quasi-Fermi level",\
        "Hole quasi-Fermi level", "Electrostatic potential",\
        "Electron density", "Hole density", "Bulk SRH recombination",\
        "Radiative recombination", "Electron current", "Hole current"]
        self.quantity.addItems(quantities)
        twoDLayout.addWidget(self.quantity)
        self.plotBtnS = QPushButton("Plot")
        self.plotBtnS.clicked.connect(self.surfacePlot)
        twoDLayout.addWidget(self.plotBtnS)
        twoDBox.setLayout(twoDLayout)
        prepare.addWidget(twoDBox)



        # Linear plot
        oneDBox = QGroupBox("Linear plot")
        oneDBox.setMaximumWidth(width)
        oneDLayout = QVBoxLayout()
        form = QFormLayout()

        # Choice between Loop values and position
        XradioLayout = QHBoxLayout()
        radio = QButtonGroup(XradioLayout)
        self.radioLoop = QRadioButton("Loop Values")
        self.radioLoop.toggled.connect(self.radioLoop_toggled)
        self.radioPos = QRadioButton("Position")
        self.radioPos.toggled.connect(self.radioPos_toggled)
        radio.addButton(self.radioLoop)
        radio.addButton(self.radioPos)
        XradioLayout.addWidget(self.radioLoop)
        XradioLayout.addWidget(self.radioPos)

        # Create the form
        self.Xdata = QLineEdit()
        form.addRow("X data", XradioLayout)
        form.addRow("", self.Xdata)
        self.quantity2 = QComboBox()
        quantities = ["Choose one", "Band diagram",\
        "Electron quasi-Fermi level", "Hole quasi-Fermi level",\
        "Electrostatic potential","Electron density",\
        "Hole density", "Bulk SRH recombination", "Radiative recombination",\
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
        wd = self.table.simulation.workDirName.text()
        paths = dialog.getOpenFileNames(self, "Upload files", wd, "(*.npz)")[0]
        for path in paths:
            self.filesList.append(path)
            path = os.path.basename(path)
            self.dataList.addItem(path)

    def remove(self):
        # remove the selected files from the list
        for i in self.dataList.selectedItems():
            idx = self.dataList.row(i)
            self.dataList.takeItem(idx)
            del self.filesList[idx]

    @slotError("bool")
    def radioLoop_toggled(self, checked):
        # copy the loop values from simulation tab into XData area
        self.Xdata.setText(self.table.simulation.loopValues.text())
        # select (ie highlight) all files in list
        for i in range(self.dataList.count()):
            item = self.dataList.item(i)
            item.setSelected(True)
        # disable some combo box rows
        for i in range(1,13):
            self.quantity2.model().item(i).setEnabled(False)
        # enable some rows
            self.quantity2.model().item(13).setEnabled(True)

    def radioPos_toggled(self):
        # give example in XData area
        settings = self.table.build.getSystemSettings()
        system = parseSettings(settings)
        if system.dimension == 1:
            self.Xdata.setText("(0,0), ({}, 0)".format(system.xpts[-1]))
        else:
            self.Xdata.setText("(x1, y1), (x2, y2)")
        # disable some combo box rows
        self.quantity2.model().item(13).setEnabled(False)
        # enable some combo box rows
        for i in range(1,13):
            self.quantity2.model().item(i).setEnabled(True)

        
    @slotError("bool")
    def surfacePlot(self, checked):
        # get system
        settings = self.table.build.getSystemSettings()
        system = parseSettings(settings)

        # get data from file
        files = [self.filesList[self.dataList.row(i)]\
                for i in self.dataList.selectedItems()
        ]
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

            # plot
            txt = self.quantity.currentText()
            self.surfaceFig.figure.clear()
            if txt == "Electron quasi-Fermi level":
                dataMap = vt * az.efn
                title = r'$\mathregular{E_{F_n}}$ [eV]'
            if txt == "Hole quasi-Fermi level":
                dataMap = vt * az.efp
                title = r'$\mathregular{E_{F_p}}$ [eV]'
            if txt == "Electrostatic potential":
                dataMap = vt * az.v
                title = r'$\mathregular{V}$ [eV]'
            if txt == "Electron density":
                dataMap = N * az.electron_density()
                title = r'n [$\mathregular{cm^{-3}}$]'
            if txt == "Hole density":
                dataMap = N * az.hole_density()
                title = r'p [$\mathregular{cm^{-3}}$]'
            if txt == "Bulk SRH recombination":
                dataMap = G * az.bulk_srh_rr()
                title = r'Bulk SRH [$\mathregular{cm^{-3}s^{-1}}$]'
            if txt == "Radiative recombination":
                dataMap = G * az.radiative_rr()
                title = r'Radiative Recomb. [$\mathregular{cm^{-3}s^{-1}}$]'
            if txt != "Electron current" and txt != "Hole current":
                plot(system, dataMap, scale=1e-4, cmap='viridis',\
                     fig=self.surfaceFig.figure, title=title)
 
            if txt == "Electron current":
                az.current_map(True, 'viridis', 1e4, fig=self.surfaceFig.figure)

            if txt == "Hole current":
                az.current_map(False, 'viridis', 1e4, fig=self.surfaceFig.figure)

            self.surfaceFig.canvas.draw()

    @slotError("bool")
    def linearPlot(self, checked):

        # check if Xdata type is selected
        if not self.radioLoop.isChecked() and not self.radioPos.isChecked():
            msg = QMessageBox()
            msg.setWindowTitle("Processing error")
            msg.setIcon(QMessageBox.Critical)
            msg.setText("No X data type chosen.")
            msg.setEscapeButton(QMessageBox.Ok)
            msg.exec_()
            return

        # get data files names
        files = [self.filesList[self.dataList.row(i)]\
                for i in self.dataList.selectedItems()
        ]
        files.sort() # so that loop values coincide with file order
        if len(files) == 0:
            return

        # test what kind of plot we are making
        exec("Xdata = {0}".format(self.Xdata.text()), globals())
        txt = self.quantity2.currentText()

        if self.radioLoop.isChecked():
            try:
                iter(Xdata)
            except TypeError:
                msg = QMessageBox()
                msg.setWindowTitle("Processing error")
                msg.setIcon(QMessageBox.Critical)
                msg.setText("The loop values expression is not iterable.")
                msg.setEscapeButton(QMessageBox.Ok)
                msg.exec_()
                return

            if len(Xdata) != len(files):
                msg = QMessageBox()
                msg.setWindowTitle("Processing error")
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Number of selected files does not match number of loop values.")
                msg.setEscapeButton(QMessageBox.Ok)
                msg.exec_()
                return

        if self.radioPos.isChecked() and not isinstance(Xdata[0], tuple):
            msg = QMessageBox()
            msg.setWindowTitle("Processing error")
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Provide two tuples defining a line as the X data.")
            msg.setEscapeButton(QMessageBox.Ok)
            msg.exec_()
            return           

 
        # get system
        settings = self.table.build.getSystemSettings()
        system = parseSettings(settings)

        # scalings
        vt = system.scaling.energy
        N  = system.scaling.density
        G  = system.scaling.generation
        J  = system.scaling.current

        # clear the figure
        self.linearFig.figure.clear()

        # loop over the files and plot
        for fdx, fileName in enumerate(files):
            data = np.load(fileName)
            az = Analyzer(system, data)

            # get sites and coordinates of a line or else
            if isinstance(Xdata[0], tuple):
                if system.dimension == 1:
                    X = system.xpts
                    sites = np.arange(system.nx, dtype=int)
                if system.dimension == 2:
                    X, sites = az.line(system, Xdata[0], Xdata[1])
                    X = X * system.scaling.length
            else:
                X = Xdata

            # get the corresponding Y data
            if txt == "Electron quasi-Fermi level":
                Ydata = vt * az.efn[sites]
                YLabel = r'$\mathregular{E_{F_n}}$ [eV]'
            if txt == "Hole quasi-Fermi level":
                Ydata = vt * az.efp[sites]
                YLabel = r'$\mathregular{E_{F_p}}$ [eV]'
            if txt == "Electrostatic potential":
                Ydata = vt * az.v[sites]
                YLabel = 'V [eV]'
            if txt == "Electron density":
                Ydata = N * az.electron_density()[sites]
                YLabel = r'n [$\mathregular{cm^{-3}}$]'
            if txt == "Hole density":
                Ydata = N * az.hole_density()[sites]
                YLabel = r'p [$\mathregular{cm^{-3}}$]'
            if txt == "Bulk SRH recombination":
                Ydata = G * az.bulk_srh_rr()[sites]
                YLabel = r'Bulk SRH [$\mathregular{cm^{-3}s^{-1}}$]'
            if txt == "Radiative recombination":
                Ydata = G * az.radiative_rr()[sites]
                YLabel = r'Radiative recombination [$\mathregular{cm^{-3}s^{-1}}$]'
            if txt == "Electron current along x":
                Ydata = J * az.electron_current(component='x')[sites] * 1e3
                YLabel = r'$\mathregular{J_{n,x}\ [mA\cdot cm^{-2}]}$'
            if txt == "Hole current along x":
                Ydata = J * az.hole_current(component='x')[sites] * 1e3
                YLabel = r'$\mathregular{J_{p,x}\ [mA\cdot cm^{-2}]}$'
            if txt == "Electron current along y":
                Ydata = J * az.electron_current(component='y')[sites] * 1e3
                YLabel = r'$\mathregular{J_{n,y}\ [mA\cdot cm^{-2}]}$'
            if txt == "Hole current along y":
                Ydata = J * az.hole_current(component='y')[sites] * 1e3
                YLabel = r'$\mathregular{J_{p,y}\ [mA\cdot cm^{-2}]}$'
            if txt == "Full steady state current":
                Ydata = J * az.full_current() * 1e3
                if system.dimension == 1:
                    YLabel = r'J [$\mathregular{mA\cdot cm^{-2}}$]'
                if system.dimension == 2:
                    YLabel = r'J [$\mathregular{mA\cdot cm^{-1}}$]'

            # plot
            if txt != "Band diagram": # everything except band diagram
                ax = self.linearFig.figure.add_subplot(111)
                if txt == "Full steady state current":
                    ax.plot(X[fdx], Ydata, 'ko')
                    ax.set_ylabel(YLabel)
                else:
                    X = X * 1e4  # set length in um
                    ax.plot(X, Ydata)
                    ax.set_ylabel(YLabel)
                    ax.set_xlabel(r'Position [$\mathregular{\mu m}$]')
            else:
                az.band_diagram((Xdata[0], Xdata[1]), fig=self.linearFig.figure)
                
            self.linearFig.canvas.draw()
