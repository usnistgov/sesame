# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

from .system_tab import BuilderBox
from .simulation_tab import Simulation
from .analysis_tab import Analysis
from .. import plotter
from .common import parseSettings, slotError

import sys
import os
os.environ['QT_API'] = 'pyqt5'
import sip
sip.setapi("QString", 2)
sip.setapi("QVariant", 2)
from PyQt5.QtGui  import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from qtconsole.rich_ipython_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from IPython import version_info
from IPython.lib import guisupport

from configparser import ConfigParser
config = ConfigParser()
config.optionxform = str
config.add_section('System')
config.add_section('Simulation')

from ast import literal_eval as ev

class Window(QMainWindow): 
    """
    Class defining the main window of the GUI.
    """
    def __init__(self):
        super(Window, self).__init__()

        self.init_ui()

    def init_ui(self):
        'init the UI'

        #============================================
        # Menu bar
        #============================================
        menuBar = self.menuBar()
        menuBar.setNativeMenuBar(False)

        # File menu
        fileMenu = menuBar.addMenu("&File")
        openAction = QAction('Open...', self)
        saveAction = QAction('Save', self)
        saveAsAction = QAction('Save as...', self)
        exitAction = QAction('Exit', self)
        fileMenu.addAction(openAction)
        fileMenu.addAction(saveAction)
        fileMenu.addAction(saveAsAction)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAction)

        # View menu
        viewMenu = menuBar.addMenu("&View")
        view1 = QAction('Lines defects', self)
        view2 = QAction('Planes defects', self)
        viewMenu.addAction(view1)
        viewMenu.addAction(view2)
        viewMenu.addSeparator()

        view3 = viewMenu.addMenu('Mobility')
        view31 = QAction('Electron', self)
        view32 = QAction('Hole', self)
        view3.addAction(view31)
        view3.addAction(view32)

        view4 = viewMenu.addMenu('Lifetime')
        view41 = QAction('Electron', self)
        view42 = QAction('Hole', self)
        view4.addAction(view41)
        view4.addAction(view42)

        # IPython menu
        ipythonMenu = menuBar.addMenu("&Console")
        ip1 = QAction('Show console', self)
        ipythonMenu.addAction(ip1)

        # actions
        openAction.triggered.connect(self.openConfig)
        saveAction.triggered.connect(self.saveConfig)
        saveAsAction.triggered.connect(self.saveAsConfig)
        exitAction.triggered.connect(self.close)
        view1.triggered.connect(lambda: self.displayPlot("lines",1))
        view2.triggered.connect(lambda: self.displayPlot("planes",1))
        view31.triggered.connect(lambda: self.displayPlot("mu_e",1))
        view32.triggered.connect(lambda: self.displayPlot("mu_h",1))
        view41.triggered.connect(lambda: self.displayPlot("tau_e",1))
        view42.triggered.connect(lambda: self.displayPlot("tau_h",1))
        ip1.triggered.connect(lambda: self.dock.show())

        #============================================
        # Window settings
        #============================================
        # Basic settings
        self.setWindowTitle('Sesame')
        # QApplication.setWindowIcon(QIcon('/home/bhg/Desktop/logo_sesame2.png'))
        # Create geomtry and center the window
        self.setGeometry(0,0,1000,700)
        windowFrame = self.frameGeometry()
        screenCenter = QDesktopWidget().availableGeometry().center()
        windowFrame.moveCenter(screenCenter)
        self.move(windowFrame.topLeft())

        # Set window tabs
        self.table = TableWidget(self)
        self.setCentralWidget(self.table)

        # Create and hide a dock for an IPython console
        self.dock = QDockWidget("  IPython console", self)
        self.ipython = IPythonWidget(self)
        self.dock.setWidget(self.ipython)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock)
        self.dock.hide()

        # Show the interface
        self.show()

    def setSystem(self, grid, doping, materials, defects, gen, param):
        """
        Fill out all fields of the interface system tab with the settings from
        the configuration file.  
        """
        
        # system tab instance 
        build = self.table.build
        
        # edit QLineEdit widgets
        build.g1.setText(grid[0])
        build.g2.setText(grid[1])
        build.g3.setText(grid[2])
        build.loc1.setText(doping[0]['location'])
        build.N1.setText(doping[0]['concentration'])
        build.loc2.setText(doping[1]['location'])
        build.N2.setText(doping[1]['concentration'])
        build.gen.setText(gen)
        build.paramName.setText(param)

        # set materials
        build.materials_list = materials
        build.matNumber = -1
        build.box.clear()
        build.loc.clear()
        for mat in materials:
            # location
            build.loc.setText(mat['location'])
            # add "material number" to combo box
            build.matNumber += 1
            build.box.addItem("Material " + str(build.matNumber+1))
            build.box.setCurrentIndex(build.matNumber)
            # fill in table with material values
            values = [mat['Nc'], mat['Nv'], mat['Eg'], mat['epsilon'],\
                      mat['mass_e'], mat['mass_h'], mat['mu_e'], mat['mu_h'],\
                      mat['Et'], mat['tau_e'], mat['tau_h'], mat['affinity'],\
                      mat['B'], mat['Cn'], mat['Cp']]
            for idx, (val, unit) in enumerate(zip(values, build.units)):
                build.table.setItem(idx,0, QTableWidgetItem(str(val)))
                item = QTableWidgetItem(unit)
                item.setFlags(Qt.ItemIsEnabled)
                build.table.setItem(idx,1, item)
            build.table.show()
            build.loc.show()
            build.lbl.show()
        # Disable remove and new buttons, enable save button
        if len(build.materials_list) > 1:
            build.removeButton.setEnabled(True)
        build.newButton.setEnabled(True)
        build.saveButton.setEnabled(False)

        # set defects properties 
        build.defects_list = defects
        build.defectNumber = -1
        build.defectBox.clear()
        build.cloc.clear()
        for defect in defects:
            # location
            build.cloc.setText(defect['location'])
            # add "defect number" to combo box
            build.defectNumber += 1
            build.defectBox.addItem("Defect " + str(build.defectNumber+1))
            build.defectBox.setCurrentIndex(build.defectNumber)
            # fill in table with defect values
            defectValues = [defect['Energy'], defect['Density'],\
                            defect['sigma_e'], defect['sigma_h'],\
                            defect['Transition']]
            for idx, (val, unit) in enumerate(zip(defectValues, build.units2)):
                build.ctable.setItem(idx,0, QTableWidgetItem(str(val)))
                item = QTableWidgetItem(unit)
                item.setFlags(Qt.ItemIsEnabled)
                build.ctable.setItem(idx,1, item)
            build.ctable.show()
            build.cloc.show()
            build.clbl.show()
        # Disable remove and new buttons, enable save button
        build.removeButton2.setEnabled(True)
        build.defectButton.setEnabled(True)
        build.saveButton2.setEnabled(False)



    def setSimulation(self, voltageLoop, loopValues, workDir, fileName, ext,\
                      BCs, ScnL, ScpL, ScnR, ScpR, precision, maxSteps,\
                      useMumps, iterative):
        """
        Fill out all fields of the interface simulation tab with the settings
        from the configuration file.  
        """
        if voltageLoop:
            self.table.simulation.voltage.setChecked(True)
        else:
            self.table.simulation.other.setChecked(True)
        self.table.simulation.loopValues.setText(loopValues)
        self.table.simulation.workDirName.setText(workDir)
        self.table.simulation.fileName.setText(fileName)
        if ext == '.npz':
            self.table.simulation.fbox.setCurrentIndex(0)
        elif ext == '.mat':
            self.table.simulation.fbox.setCurrentIndex(1)
        if BCs:
            self.table.simulation.periodic.setChecked(True)
        else:
            self.table.simulation.hardwall.setChecked(True)
        self.table.simulation.g4.setText(ScnL)
        self.table.simulation.g5.setText(ScpL)
        self.table.simulation.g6.setText(ScnR)
        self.table.simulation.g7.setText(ScpR)
        self.table.simulation.algoPrecision.setText(precision)
        self.table.simulation.algoSteps.setText(maxSteps)
        if useMumps:
            self.table.simulation.yesMumps.setChecked(True)
        else:
            self.table.simulation.noMumps.setChecked(True)
        if iterative:
            self.table.simulation.yesIterative.setChecked(True)
        else:
            self.table.simulation.noIterative.setChecked(True)
        
    def openConfig(self):
        """
        Open and read the configuration of the interface system and analysis
        tabs. This file must end with extension .ini.
        """
        self.cfgFile = QFileDialog.getOpenFileName(self, 'Open File', '',\
                        "(*.ini)")[0]
        if self.cfgFile == '':
            return

        with open(self.cfgFile, 'r') as f:
            try:
                config.read(self.cfgFile)
            except:
                f.close()
                msg = QMessageBox()
                msg.setWindowTitle("Processing error")
                msg.setIcon(QMessageBox.Critical)
                msg.setText("The file could not be read.")
                msg.setEscapeButton(QMessageBox.Ok)
                msg.exec_()
                return

            grid = config.get('System', 'Grid')
            doping = config.get('System', 'Doping')
            materials = config.get('System', 'Materials')
            defects = config.get('System', 'Defects')
            gen, param = config.get('System', 'Generation rate'),\
                         config.get('System', 'Generation parameter')
            self.setSystem(ev(grid), ev(doping), ev(materials), ev(defects),\
                           gen, param)

            voltageLoop = config.getboolean('Simulation', 'Voltage loop')
            loopValues = config.get('Simulation', 'Loop values')
            workDir = config.get('Simulation', 'Working directory')
            fileName = config.get('Simulation', 'Simulation name')
            ext = config.get('Simulation', 'Extension')
            BCs = config.getboolean('Simulation', 'Transverse boundary conditions')
            ScnL = config.get('Simulation', 'Electron recombination velocity in 0')
            ScpL = config.get('Simulation', 'Hole recombination velocity in 0')
            ScnR = config.get('Simulation', 'Electron recombination velocity in L')
            ScpR = config.get('Simulation', 'Hole recombination velocity in L')
            precision = config.get('Simulation', 'Newton precision')
            maxSteps = config.get('Simulation', 'Maximum steps')
            useMumps = config.getboolean('Simulation', 'Use Mumps')
            iterative = config.getboolean('Simulation', 'Iterative solver')
            self.setSimulation(voltageLoop, loopValues, workDir, fileName, \
                               ext, BCs, ScnL, ScpL, ScnR, ScpR, precision,\
                               maxSteps, useMumps, iterative)

            f.close()

    def saveAsConfig(self):
        self.cfgFile = QFileDialog.getSaveFileName(self, 'Save File', '.ini', \
                        "(*.ini)")[0]
        if self.cfgFile != '':
            # append extension if changed
            if self.cfgFile[-4:] != '.ini':
                self.cfgFile += '.ini'
            self.saveConfig()

    def saveConfig(self):
        """
        Save the configuration of the interface system and analysis
        tabs in a file with extension .ini.
        """

        if not hasattr(self, 'cfgFile'):
            self.saveAsConfig()
        elif self.cfgFile == '':
            return
        else:
            build = self.table.build
            simu = self.table.simulation

            grid = [build.g1.text(), build.g2.text(), build.g3.text()]
            d = [{'location': build.loc1.text(), 'concentration': build.N1.text()},
                 {'location': build.loc2.text(), 'concentration': build.N2.text()}]
            mat = build.materials_list
            defects = build.defects_list
            gen, param = build.gen.text(), build.paramName.text()

            with open(self.cfgFile, 'w') as f:
                config.set('System', 'Grid', str(grid))
                config.set('System', 'Doping', str(d))
                config.set('System', 'Materials', str(mat))
                config.set('System', 'Defects', str(defects))
                config.set('System', 'Generation rate', gen)
                config.set('System', 'Generation parameter', param)

                config.set('Simulation', 'Voltage loop',\
                                            str(simu.voltage.isChecked()))
                config.set('Simulation', 'Loop values', simu.loopValues.text())
                config.set('Simulation', 'Working directory', simu.workDirName.text())
                config.set('Simulation', 'Simulation name', simu.fileName.text())
                config.set('Simulation', 'Extension', simu.fbox.currentText())
                config.set('Simulation', 'Transverse boundary conditions',\
                                                str(simu.periodic.isChecked()))
                config.set('Simulation', 'Electron recombination velocity in 0',\
                                            simu.g4.text())
                config.set('Simulation', 'Hole recombination velocity in 0',\
                                            simu.g5.text())
                config.set('Simulation', 'Electron recombination velocity in L',\
                                            simu.g6.text())
                config.set('Simulation', 'Hole recombination velocity in L',\
                                            simu.g7.text())
                config.set('Simulation', 'Newton precision', simu.algoPrecision.text())
                config.set('Simulation', 'Maximum steps', simu.algoSteps.text())
                config.set('Simulation', 'Use Mumps', str(simu.yesMumps.isChecked()))
                config.set('Simulation', 'Iterative solver',\
                                            str(simu.yesIterative.isChecked()))

                config.write(f)
                f.close()

    @slotError()
    def displayPlot(self, prop, checked):
        """
        Access to plotter routines to visualize some basic system settings in 2D
        to make sure the regions are defined as envisioned.
        """
        settings = self.table.build.getSystemSettings()
        system = parseSettings(settings)
        if prop == "mu_e":
            plotter.plot(system, system.mu_e)
        elif prop == "mu_h":
            plotter.plot(system, system.mu_h)
        elif prop == "tau_e":
            plotter.plot(system, system.tau_e)
        elif prop == "tau_h":
            plotter.plot(system, system.tau_h)
        elif prop == "lines":
            plotter.plot_line_defects(system)
        elif prop == "planes":
            plotter.plot_plane_defects(system)

        
class QIPythonWidget(RichJupyterWidget):
    """ 
    Convenience class for the definition of a live IPython console widget. We
    replace the standard banner using the sesameBanner argument.
    """ 
    
    def __init__(self, *args, **kwargs):
        super(QIPythonWidget, self).__init__(*args,**kwargs)
        # banners printed at the top of the shell
        self.banner = ''
        banner1 = 'Python ' + sys.version.replace('\n', '')
        banner2 = 'IPython ' + ".".join(map(str, version_info[:3]))
        # create qt console kernel
        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel_manager.kernel.shell.banner1 = banner1
        kernel_manager.kernel.shell.banner2 = banner2
        kernel_manager.kernel.gui = 'qt'
        self.kernel_client = kernel_client = self._kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
            guisupport.get_app_qt5().exit()            
        self.exit_requested.connect(stop)

    def pushVariables(self,variableDict):
        # Given a dictionary containing name / value pairs, push those
        # variables to the IPython console widget 
        self.kernel_manager.kernel.shell.push(variableDict)
    def clearTerminal(self):
        # Clears the terminal
        self._control.clear()    
    def printText(self,text):
        # Prints some plain text to the console
        self._append_plain_text(text)        
    def executeCommand(self,command):
        # Execute a command in the frame of the console widget
        self._execute(command,False)


class IPythonWidget(QWidget):
    """ 
    GUI widget including an IPython console inside a vertical layout. 
    """ 
    
    def __init__(self, parent=None):
        super(IPythonWidget, self).__init__(parent)
        console = QIPythonWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(console)        

class TableWidget(QWidget):
    """
    Definition of the three tabs that make the GUI: system, simulation,
    analysis.
    """
    def __init__(self, parent):
        super(TableWidget, self).__init__(parent)

        self.parent = parent

        self.layout = QHBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tabs.resize(300,200) 

        # Add tabs
        self.tabs.addTab(self.tab1,"System")
        self.tabs.addTab(self.tab2,"Simulation")
        self.tabs.addTab(self.tab3,"Analysis")
        # Add tabs to widget        
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        #============================================
        #  tab1: system parameters
        #============================================
        self.tab1Layout = QHBoxLayout(self.tab1)
        self.build = BuilderBox(self)
        self.tab1Layout.addWidget(self.build)

        #============================================
        #  tab2: run the simulation
        #============================================
        self.tab2Layout = QHBoxLayout(self.tab2)
        self.simulation = Simulation(self)
        self.tab2Layout.addWidget(self.simulation)

        #============================================
        #  tab3: analyze simulation results
        #============================================
        self.tab3Layout = QHBoxLayout(self.tab3)
        self.analysis = Analysis(self)
        self.tab3Layout.addWidget(self.analysis)
