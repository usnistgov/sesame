from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
 
import sys
import numpy as np 
import logging
from ast import literal_eval as ev
from io import StringIO

from ..solvers import IVcurve
from .common import parseSettings, slotError
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
    def __init__(self, parent):
        super(Simulation, self).__init__(parent)

        self.mainWindow = parent

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
        self.vlayout = QVBoxLayout()

        #######  Basics
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

        ######  Boundary conditions
        self.BCbox = QGroupBox("Boundary conditions")
        BCform = QFormLayout()
        self.BCbox.setLayout(BCform)
        self.vlayout.addWidget(self.BCbox)

        # contacts BCs
        contactLayout = QHBoxLayout()
        contact = QButtonGroup(contactLayout)
        self.dirichlet = QRadioButton("Dirichlet")
        self.dirichlet.setChecked(True)
        self.neumann = QRadioButton("Neumann")
        contact.addButton(self.dirichlet)
        contact.addButton(self.neumann)
        contactLayout.addWidget(self.dirichlet)
        contactLayout.addWidget(self.neumann)
        BCform.addRow("Contacts boundary conditions", contactLayout)

        # contacts surface recombination velocities
        self.g4 = QLineEdit("", self)
        self.g5 = QLineEdit("", self)
        self.g6 = QLineEdit("", self)
        self.g7 = QLineEdit("", self)
        BCform.addRow("Electron recombination velocity in x=0 [m/s]", self.g4)
        BCform.addRow("Hole recombination velocity in x=0 [m/s]", self.g5)
        BCform.addRow("Electron recombination velocity in x=L [m/s]", self.g6)
        BCform.addRow("Hole recombination velocity in x=L [m/s]", self.g7)

        # transverse BC
        tbcLayout = QHBoxLayout()
        tbc = QButtonGroup(contactLayout)
        self.periodic = QRadioButton("Periodic")
        self.hardwall = QRadioButton("Hardwall")
        tbc.addButton(self.periodic)
        tbc.addButton(self.hardwall)
        tbcLayout.addWidget(self.periodic)
        tbcLayout.addWidget(self.hardwall)
        BCform.addRow("Transverse boundary conditions", tbcLayout)

        ######  Advanced settings
        self.algoBox = QGroupBox("Algorithm settings")
        self.form2 = QFormLayout()

        # algo tol, maxiter, 
        self.algoPrecision = QLineEdit("1e-6", self)
        self.algoSteps = QLineEdit("100", self)
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

        self.algoBox.setLayout(self.form2)
        self.vlayout.addWidget(self.algoBox)
        self.tabLayout.addLayout(self.vlayout)

        #===============================================
        #  Run simulation
        #===============================================
        self.vlayout2 = QVBoxLayout()

        self.simBox = QGroupBox("Simulation log")
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
    
    def browse(self):
        dialog = QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Select Folder")
        self.workDirName.setText(folder_path)

    def getSolverSettings(self):
        # loopValues
        loopValues = ev(self.loopValues.text())
        loopValues = np.asarray(loopValues)
        # simulation name
        simName = self.workDirName.text() + self.fileName.text()
        # extension
        extension = self.fbox.currentText()

        # contacts BCs
        if self.dirichlet.isChecked():
            contacts = "Dirichlet"
        else:
            contacts = "Neumann"
        # transverse BCs
        if self.periodic.isChecked():
            BCs = True
        else:
            BCs = False
        ScnL, ScpL = float(self.g4.text()), float(self.g5.text())
        ScnR, ScpR = float(self.g6.text()), float(self.g7.text())

        # precision
        precision = float(self.algoPrecision.text())
        # max steps
        steps = int(self.algoSteps.text())
        # mumps
        useMumps = self.yesMumps.isChecked()
        # internal iterative solver
        iterative = self.yesIterative.isChecked()

        settings = [loopValues, simName, extension, BCs, contacts, \
                    [ScnL, ScpL, ScnR, ScpR], precision, steps,\
                    useMumps, iterative]
        return settings


    @slotError("bool")
    def run(self, checked):

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
                return

        # get system settings and build system without generation
        settings = self.mainWindow.table.build.getSystemSettings()
        system = parseSettings(settings)
        generation, paramName = settings['gen']

        # get solver settings
        solverSettings = self.getSolverSettings()

        # define a thread in which to run the simulation
        self.thread = QThread(self)
        self.thread.start()

        # add worker to thread and run simulation
        self.simulation = SimulationWorker(loop, system, solverSettings, generation, paramName)
        self.simulation.moveToThread(self.thread)
        self.simulation.start.connect(self.simulation.run)
        self.simulation.start.emit("hello")

    @pyqtSlot(str)
    def displayMessage(self, message):
        if message != "\n":        
            self.logWidget.appendPlainText(message)

    def stop(self):
        if self.thread.isRunning():
            self.thread.terminate()
            self.thread.wait()
            self.logger.critical("****** Calculation interrupted manually ******")
