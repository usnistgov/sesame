from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
 
import sys
import numpy as np 
import logging
from ast import literal_eval as ev

from ..solvers import IVcurve
from .common import parseSettings, slotError
from .sim import run_sim


class Simulation(QWidget):
    def __init__(self, parent):
        super(Simulation, self).__init__(parent)

        self.mainWindow = parent

        self.tabLayout = QHBoxLayout()

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
        self.fileName = QLineEdit()
        self.form1.addRow("Loop values", self.loopValues)
        self.form1.addRow("Output file name", self.fileName)
        self.fbox = QComboBox()
        self.fbox.addItem(".npz")
        self.fbox.addItem(".mat")
        self.form1.addRow("File extension", self.fbox)

        self.outputBox.setLayout(self.form1)
        self.vlayout.addWidget(self.outputBox)

        ######  Advanced settings
        self.algoBox = QGroupBox("Algorithm settings")

        self.form2 = QFormLayout()

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
        self.form1.addRow("Contacts boundary conditions", contactLayout)

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
        # self.brun.adjustSize()
        # self.bstop.adjustSize()
        self.buttons.addWidget(self.brun)
        # self.buttons.addWidget(self.bstop)
        self.logLayout.addLayout(self.buttons)

        # log
        self.log = LogWidget()
        self.logLayout.addWidget(self.log)

        self.vlayout2.addWidget(self.simBox)

        self.tabLayout.addLayout(self.vlayout2)
        self.setLayout(self.tabLayout)

    def getSolverSettings(self):
        # loopValues
        loopValues = ev(self.loopValues.text())
        loopValues = np.asarray(loopValues)
        # simulation name
        simName = self.fileName.text()
        # extension
        extension = self.fbox.currentText()
        # contacts BCs
        if self.dirichlet.isChecked():
            contacts = "Dirichlet"
        else:
            contacts = "Neumann"
        # precision
        precision = float(self.algoPrecision.text())
        # max steps
        steps = int(self.algoSteps.text())
        # mumps
        useMumps = self.yesMumps.isChecked()
        # internal iterative solver
        iterative = self.yesIterative.isChecked()
        # transverse BCs
        BCs = self.mainWindow.entry.get_bcs()
        if BCs == 'Periodic':
            BCs = True
        else:
            BCs = False

        settings = [loopValues, simName, extension, BCs, contacts, precision, steps,\
                    useMumps, iterative]
        return settings

    @slotError("bool")
    def run(self, checked):
        # get system settings and build system without generation
        settings = self.mainWindow.table.settingsBox.get_settings()
        system = parseSettings(settings)
        generation, paramName = settings['gen']

        # get solver settings
        solverSettings = self.getSolverSettings()

        # loop over voltages
        if self.voltage.isChecked():
            run_sim("voltage",system, solverSettings, generation, paramName)

        # loop over generation rates
        elif self.other.isChecked():
            run_sim("generation",system, solverSettings, generation, paramName)
            

class StreamToLogger():
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

class QPlainTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()

        self.widget = QPlainTextEdit(parent)
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)
 
class LogWidget(QWidget):
    def __init__(self):
        super(LogWidget, self).__init__()

        self.layout = QHBoxLayout(self)

        log_handler = QPlainTextEditLogger(self)
        log_handler.setFormatter(\
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_handler)
        logging.getLogger().setLevel(logging.INFO)

        # Send stdout to the logger
        sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)
        # sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)

        self.layout.addWidget(log_handler.widget)
