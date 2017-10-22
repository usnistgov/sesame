from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
 
import sys
import numpy as np 
import logging
from ast import literal_eval as ev

from ..solvers import IVcurve
from .makeSystem import parseSettings


class Simulation(QWidget):
    def __init__(self, parent):
        super(Simulation, self).__init__(parent)

        self.mainWindow = parent

        self.tabLayout = QHBoxLayout()

        #===============================================
        #  Settings boxes
        #===============================================
        self.vlayout = QVBoxLayout()

        # Voltages and files
        self.outputBox = QGroupBox("Voltages and output files")

        self.form1 = QFormLayout()
        self.voltages = QLineEdit("(0, 0.05, 0.1)", self)
        self.fileName = QLineEdit()
        self.form1.addRow("Applied voltages", self.voltages)
        self.form1.addRow("Output file name", self.fileName)
        self.fbox = QComboBox()
        self.fbox.addItem(".npz")
        self.fbox.addItem(".mat")
        self.form1.addRow("File extension", self.fbox)

        self.outputBox.setLayout(self.form1)
        self.vlayout.addWidget(self.outputBox)

        # Advanced settings
        self.algoBox = QGroupBox("Algorithm settings")

        self.form2 = QFormLayout()
        self.algoPrecision = QLineEdit("1e-6", self)
        self.algoSteps = QLineEdit("100", self)
        self.form2.addRow("Algorithm precision", self.algoPrecision)
        self.form2.addRow("Maximum steps", self.algoSteps)

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
        # voltages
        vapp = ev(self.voltages.text())
        appliedVoltages = np.asarray(vapp)
        # simulation name
        simName = self.fileName.text()
        # extension
        extension = self.fbox.currentText()
        # precision
        precision = float(self.algoPrecision.text())
        # max steps
        steps = int(self.algoSteps.text())
        # mumps
        useMumps = self.yesMumps.isChecked()
        # internal iterative solver
        iterative = self.yesIterative.isChecked()

        settings = [appliedVoltages, simName, extension, precision, steps,\
                    useMumps, iterative]
        return settings

    def run(self):
        # get solver settings
        voltages, simName, fmt, tol, steps,\
                    useMumps, iterative = self.getSolverSettings()
        # get system and boundary conditions
        settings = self.mainWindow.table.settingsBox.get_settings()
        system = parseSettings(settings)
        BCs = self.mainWindow.entry.get_bcs()
        if BCs == 'Periodic':
            BCs = True
        else:
            BCs = False

        # run simulation
        try:
            IVcurve(system, voltages, simName, tol=tol, periodic_bcs=BCs,\
                maxiter=steps, verbose=True, use_mumps=useMumps,\
                iterative=iterative, inner_tol=1e-6, htp=1, fmt=fmt)
            print("Calculation done")
        except:
            pass


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
        # sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)
        # sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)

        self.layout.addWidget(log_handler.widget)
