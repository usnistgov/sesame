# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
 
import os
import sys
import numpy as np 
import logging
from ast import literal_eval as ev
from io import StringIO
import traceback

from .common import parseSettings
from .sim import SimulationWorker


class PrimitiveSignals(QObject):
    signal_str = pyqtSignal(str)
    def __init__(self):
        QObject.__init__(self)


class logBuffer(StringIO):
    def __init__(self):
        self.output = PrimitiveSignals()

    def write(self, message):
        self.output.signal_str.emit(message)


class Simulation(QWidget):
    """
    UI of the simulation tab with simulation and solver settings, and logging to
    follow the output of the solver.
    """
    def __init__(self, parent):
        super(Simulation, self).__init__(parent)

        self.tabsTable = parent

        self.tabLayout = QHBoxLayout()

        self.logBuffer = logBuffer()
        self.logBuffer.output.signal_str.connect(self.displayMessage)
        logFormatter = logging.Formatter('%(levelname)s: %(message)s')
        logHandler = logging.StreamHandler(self.logBuffer)
        logHandler.setFormatter(logFormatter)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logHandler)

        #===============================================
        #  Settings boxes
        #===============================================
        self.vlayout = QVBoxLayout() # main
        self.hlayout = QHBoxLayout() # secondary
        self.tabLayout.addLayout(self.vlayout)

        #===============================================
        #  Basics
        #===============================================
        self.outputBox = QGroupBox("Basic settings")

        self.form1 = QFormLayout()

        # loop over
        loopLayout = QHBoxLayout()
        loop = QButtonGroup(loopLayout)
        self.voltage = QRadioButton("Voltages")
        self.other = QRadioButton("Generation rates")
        loop.addButton(self.voltage)
        loop.addButton(self.other)
        loopLayout.addWidget(self.voltage)
        loopLayout.addWidget(self.other)
        self.form1.addRow("Loop over", loopLayout)

        # loop values, file name, extension
        self.loopValues = QLineEdit("", self)

        self.workDir = QHBoxLayout()
        self.workDirName = QLineEdit()
        self.browseBtn = QPushButton("Browse...")
        self.browseBtn.clicked.connect(self.browse)
        self.workDir.addWidget(self.workDirName)
        self.workDir.addWidget(self.browseBtn)

        self.fileLayout = QHBoxLayout()
        self.fileName = QLineEdit()
        self.fbox = QComboBox()
        self.fbox.addItems([".npz", ".mat"])
        self.fileLayout.addWidget(self.fileName)
        self.fileLayout.addWidget(self.fbox)

        self.form1.addRow("Loop values", self.loopValues)
        self.form1.addRow("Working directory", self.workDir)
        self.form1.addRow("Output file name", self.fileLayout)

        self.outputBox.setLayout(self.form1)
        self.vlayout.addWidget(self.outputBox)

        #===============================================
        #  Boundary conditions
        #===============================================
        self.BCbox = QGroupBox("Boundary conditions")
        BCform = QFormLayout()
        self.BCbox.setLayout(BCform)
        self.hlayout.addWidget(self.BCbox)

        # contacts BCs
        contactLayoutL = QHBoxLayout()
        contactL = QButtonGroup(contactLayoutL)
        self.L_Ohmic = QRadioButton("Ohmic")
        self.L_Ohmic.setChecked(True)
        self.L_Schottky = QRadioButton("Schottky")
        self.L_Neutral = QRadioButton("Neutral")
        contactL.addButton(self.L_Ohmic)
        contactL.addButton(self.L_Schottky)
        contactL.addButton(self.L_Neutral)
        contactLayoutL.addWidget(self.L_Ohmic)
        contactLayoutL.addWidget(self.L_Schottky)
        contactLayoutL.addWidget(self.L_Neutral)
        BCform.addRow("Contact boundary conditions at x=0", contactLayoutL)

        self.L_Schottky.toggled.connect(self.L_Schottky_toggled)
        self.L_Ohmic.toggled.connect(self.L_Ohmic_toggled)
        self.L_Neutral.toggled.connect(self.L_Ohmic_toggled)

        # contacts surface recombination velocities
        self.g4 = QLineEdit("1e5", self)
        self.g5 = QLineEdit("1e5", self)
        self.g6 = QLineEdit("1e5", self)
        self.g7 = QLineEdit("1e5", self)
        self.g8 = QLineEdit("", self)
        self.g9 = QLineEdit("", self)
        self.g8.setDisabled(True)
        self.g9.setDisabled(True)
        BCform.addRow("Electron recombination velocity in x=0 [cm/s]", self.g4)
        BCform.addRow("Hole recombination velocity in x=0 [cm/s]", self.g5)
        BCform.addRow("Metal work function [eV]", self.g8)

        contactLayoutR = QHBoxLayout()
        contactR = QButtonGroup(contactLayoutR)
        self.R_Ohmic = QRadioButton("Ohmic")
        self.R_Ohmic.setChecked(True)
        self.R_Schottky = QRadioButton("Schottky")
        self.R_Neutral = QRadioButton("Neutral")
        contactR.addButton(self.R_Ohmic)
        contactR.addButton(self.R_Schottky)
        contactR.addButton(self.R_Neutral)
        contactLayoutR.addWidget(self.R_Ohmic)
        contactLayoutR.addWidget(self.R_Schottky)
        contactLayoutR.addWidget(self.R_Neutral)
        BCform.addRow("Contact boundary conditions at x=L", contactLayoutR)
        BCform.addRow("Electron recombination velocity in x=L [cm/s]", self.g6)
        BCform.addRow("Hole recombination velocity in x=L [cm/s]", self.g7)
        BCform.addRow("Metal work function [eV]", self.g9)

        self.R_Schottky.toggled.connect(self.R_Schottky_toggled)
        self.R_Ohmic.toggled.connect(self.R_Ohmic_toggled)
        self.R_Neutral.toggled.connect(self.R_Ohmic_toggled)

        # transverse BC
        tbcLayout = QHBoxLayout()
        tbc = QButtonGroup(contactLayoutL)
        self.periodic = QRadioButton("Periodic")
        self.hardwall = QRadioButton("Hardwall")
        tbc.addButton(self.periodic)
        tbc.addButton(self.hardwall)
        self.periodic.setChecked(True)
        tbcLayout.addWidget(self.periodic)
        tbcLayout.addWidget(self.hardwall)
        BCform.addRow("Transverse boundary conditions", tbcLayout)

        #===============================================
        #  Advanced settings
        #===============================================
        self.algoBox = QGroupBox("Algorithm settings")
        self.form2 = QFormLayout()

        # G ramp, algo tol, maxiter, 
        self.ramp = QSpinBox()
        self.ramp.singleStep()
        self.algoPrecision = QLineEdit("1e-6", self)
        self.algoSteps = QSpinBox(self)
        self.algoSteps.setMinimum(1)
        self.algoSteps.setMaximum(1000)
        self.algoSteps.setValue(100)
        self.form2.addRow("Generation ramp", self.ramp)
        self.form2.addRow("Algorithm precision", self.algoPrecision)
        self.form2.addRow("Maximum steps", self.algoSteps)

        # mumps yes or no
        self.radioLayout = QHBoxLayout()
        mumps_choice = QButtonGroup(self.radioLayout)
        self.yesMumps = QRadioButton("Yes")
        self.noMumps = QRadioButton("No")
        self.noMumps.setChecked(True)
        mumps_choice.addButton(self.yesMumps)
        mumps_choice.addButton(self.noMumps)
        self.radioLayout.addWidget(self.yesMumps)
        self.radioLayout.addWidget(self.noMumps)
        self.form2.addRow("Mumps library", self.radioLayout)

        # iterative solver yes or no
        self.radioLayout2 = QHBoxLayout()
        iterative_choice = QButtonGroup(self.radioLayout2)
        self.yesIterative = QRadioButton("Yes")
        self.noIterative = QRadioButton("No")
        self.noIterative.setChecked(True)
        iterative_choice.addButton(self.yesIterative)
        iterative_choice.addButton(self.noIterative)
        self.radioLayout2.addWidget(self.yesIterative)
        self.radioLayout2.addWidget(self.noIterative)
        self.form2.addRow("Iterative solver", self.radioLayout2)
        self.iterPrecision = QLineEdit("1e-6", self)
        self.form2.addRow("Iterative solver precision", self.iterPrecision)
        self.htpy = QSpinBox(self)
        self.htpy.setMinimum(1)
        self.form2.addRow("Newton homotopy", self.htpy)

        self.algoBox.setLayout(self.form2)
        self.hlayout.addWidget(self.algoBox)
        self.vlayout.addLayout(self.hlayout)

        #===============================================
        #  Run simulation
        #===============================================
        self.vlayout2 = QVBoxLayout()

        self.simBox = QGroupBox("Simulation log")
        self.simBox.setMinimumWidth(300)
        self.logLayout = QVBoxLayout()
        self.simBox.setLayout(self.logLayout)

        # Run and stop buttons
        self.buttons = QHBoxLayout()
        self.brun = QPushButton("Run simulation")
        self.bstop = QPushButton("Stop simulation")
        self.brun.clicked.connect(self.run)
        self.bstop.clicked.connect(self.stop)
        self.buttons.addWidget(self.brun)
        self.buttons.addWidget(self.bstop)
        self.logLayout.addLayout(self.buttons)

        # log
        self.logWidget = QPlainTextEdit(self)
        self.logWidget.setReadOnly(True)
        self.logLayout.addWidget(self.logWidget)

        self.vlayout2.addWidget(self.simBox)

        self.tabLayout.addLayout(self.vlayout2)
        self.setLayout(self.tabLayout)

    def L_Schottky_toggled(self):
        #  enable Metal work function input
        self.g8.setEnabled(True)

    def L_Ohmic_toggled(self):
        # diable metal work function input
        self.g8.setDisabled(True)

    def R_Schottky_toggled(self):
        #  enable Metal work function input
        self.g9.setEnabled(True)

    def R_Ohmic_toggled(self):
        # diable metal work function input
        self.g9.setDisabled(True)

    def browse(self):
        dialog = QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Select Folder")
        self.workDirName.setText(folder_path + '/')

    def getLoopValues(self):
        exec("val = {0}".format(self.loopValues.text()), globals())
        try:
            values = [v for v in val]
            return values
        except TypeError:
            msg = QMessageBox()
            msg.setWindowTitle("Processing error")
            msg.setIcon(QMessageBox.Critical)
            msg.setText("The loop values expression is not iterable.")
            msg.setEscapeButton(QMessageBox.Ok)
            msg.exec_()
            return

    def getSolverSettings(self):
        # loopValues
        loopValues = self.getLoopValues()
        # simulation name
        simName = self.workDirName.text() + self.fileName.text()
        # extension
        extension = self.fbox.currentText()

        # contacts BCs
        if self.L_Ohmic.isChecked():
            contactL = "Ohmic"
            phiL = ''
        elif self.L_Schottky.isChecked():
            contactL = "Schottky"
            phiL = float(self.g8.text())
        elif self.L_Neutral.isChecked():
            contactL = "Neutral"
            phiL = ''

        if self.R_Ohmic.isChecked():
            contactR = "Ohmic"
            phiR = ''
        elif self.R_Schottky.isChecked():
            contactR = "Schottky"
            phiR = float(self.g9.text())
        elif self.R_Neutral.isChecked():
            contactR = "Neutral"
            phiR = ''

        # transverse BCs
        if self.periodic.isChecked():
            BCs = True
        else:
            BCs = False
        # contacts recombination velocities
        ScnL, ScpL = float(self.g4.text()), float(self.g5.text())
        ScnR, ScpR = float(self.g6.text()), float(self.g7.text())

        # generation ramp
        ramp = self.ramp.value()
        # precision
        precision = float(self.algoPrecision.text())
        # max steps
        steps = self.algoSteps.value()
        # mumps
        useMumps = self.yesMumps.isChecked()
        # internal iterative solver
        iterative = self.yesIterative.isChecked()
        # iterative solver precision
        iterPrec = float(self.iterPrecision.text())
        # Newton homotopy
        htpy = self.htpy.value()

        contacts_bcs = [contactL, contactR]
        settings = [loopValues, simName, extension, BCs, contacts_bcs,\
                    [phiL, phiR], [ScnL, ScpL, ScnR, ScpR], precision, steps,\
                    useMumps, iterative, ramp, iterPrec, htpy]
        return settings

    def run(self, checked):
        # Clear the list of data files already uploaded in simulation tab
        self.tabsTable.analysis.filesList.clear()
        self.tabsTable.analysis.dataList.clear()

        # Make sure a type of simulation was chosen
        loop = ""
        while(loop == ""):
            # loop over voltages
            if self.voltage.isChecked():
                loop = "voltage"
            # loop over generation rates
            elif self.other.isChecked():
                loop = "generation"
            else:
                msg = QMessageBox()
                msg.setWindowTitle("Processing error")
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Choose what to loop over: voltages or generation rates.")
                msg.setEscapeButton(QMessageBox.Ok)
                msg.exec_()
                self.brun.setEnabled(True)
                return

        # get system settings and build system without generation
        try:
            settings = self.tabsTable.build.getSystemSettings()
            system = parseSettings(settings)
            generation, paramName = settings['gen']

            # get solver settings
            solverSettings = self.getSolverSettings()

            # define a thread in which to run the simulation
            self.thread = QThread(self)

            # add worker to thread and run simulation
            self.simulation = SimulationWorker(loop, system, solverSettings,\
                                                generation, paramName)
            self.simulation.moveToThread(self.thread)
            self.simulation.simuDone.connect(self.thread_cleanup)
            self.simulation.newFile.connect(self.updateDataList)
            self.thread.started.connect(self.simulation.run)
            # Disable run button
            self.brun.setEnabled(False)
            self.thread.start()
        except Exception:
            p = traceback.format_exc()
            # Dialog box
            msg = QMessageBox()
            msg.setWindowTitle("Processing error")
            msg.setIcon(QMessageBox.Critical)
            msg.setText("An error occurred when processing your settings.")
            msg.setDetailedText(p)
            msg.setEscapeButton(QMessageBox.Ok)
            msg.exec_()
            # re enable run button
            self.brun.setEnabled(True)
            return

    @pyqtSlot(str)
    def displayMessage(self, message):
        if message != "\n":        
            self.logWidget.appendPlainText(message)

    def stop(self):
        self.brun.setEnabled(True)
        if self.thread.isRunning():
            self.simulation.abortSim()
            self.thread_cleanup()
            self.logger.critical("****** Calculation interrupted manually ******")

    def thread_cleanup(self):
        self.brun.setEnabled(True)
        self.thread.quit()
        self.thread.wait()

    @pyqtSlot(str)
    def updateDataList(self, name):
        self.tabsTable.analysis.filesList.append(name)
        name = os.path.basename(name)
        self.tabsTable.analysis.dataList.addItem(name)

