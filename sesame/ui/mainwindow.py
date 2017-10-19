from .system_tab import *
from .simulation_tab import Simulation
from .analysis_tab import Analysis

import os
os.environ['QT_API'] = 'pyqt5'
import sip
sip.setapi("QString", 2)
sip.setapi("QVariant", 2)
from PyQt5.QtGui  import *
from PyQt5.QtWidgets import *
# Import the console machinery from ipython
from qtconsole.rich_ipython_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport



class EntryDialog(QDialog):
    def __init__(self):
        super(EntryDialog, self).__init__()

        self.formGroupBox = QGroupBox("Device symmetries")
        layout = QFormLayout()
        self.dimBox = QComboBox()
        self.dimBox.addItems(["Choose one", "1D", "2D", "3D"])
        layout.addRow(QLabel("Dimensionality:"), self.dimBox)
        self.bcBox = QComboBox()
        self.bcBox.addItems(["Choose one", "Periodic", "Hardwall"])
        layout.addRow(QLabel("Boundary conditions:"), self.bcBox)
        self.formGroupBox.setLayout(layout)

        self.resize(400, 170)
        self.setWindowTitle('New system - Sesame')
 
        buttonBox = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
 
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)

        self.setWindowModality(Qt.ApplicationModal)

    def reject(self):
        exit(1)
        
    def get_dimension(self):
        if self.dimBox.currentText() == "1D":
            return 1
        elif self.dimBox.currentText() == "2D":
            return 2
        elif self.dimBox.currentText() == "3D":
            return 3

    def get_bcs(self):
        return self.bcBox.currentText()

class Window(QWidget): 
    def __init__(self):
        super(Window, self).__init__()

        self.init_ui()

    def init_ui(self):
        'init the UI'

        self.entry = EntryDialog()

        # Split the window: top with tabs, bottom with console
        splitter = QSplitter(Qt.Vertical)
        self.layout = QVBoxLayout()
        self.layout.addWidget(splitter)
        self.setLayout(self.layout)

        # Top with tabs
        self.table = TableWidget(self)
        splitter.addWidget(self.table)

        # Bottom window: Set up logging to use your widget as a handler
        self.ipython = IPythonWidget(self)
        splitter.addWidget(self.ipython)

        self.setWindowTitle('Sesame')
        # self.setGeometry(0,0, 400,400)
        # self.show()
        self.showMaximized()

        self.entry.exec_()
        dimension = self.entry.get_dimension()
        BCs = self.entry.get_bcs()
        if not dimension in [1, 2, 3]:
            print("Choose a dimensionality value")
            self.entry.exec_()
        if not BCs in ['Periodic', 'Hardwall']:
            print("Choose boundary conditions ")
            self.entry.exec_()
        if dimension == 1 and BCs == "Periodic":
            print("Periodic boundary conditions cannot be applied in 1D")
            self.entry.exec_()

        self.table.settingsBox.make_stack(dimension)

    def get_data(self):
        dimension = self.entry.get_dimension()
        BCs = self.entry.get_BCs()
        data = {'dimension': dimension, 'BCs': BCs}
        return data
        
class QIPythonWidget(RichJupyterWidget):
    """ Convenience class for a live IPython console widget. We can replace the
    standard banner using the customBanner argument
    """ 
    
    def __init__(self,customBanner=None,*args,**kwargs):
        if not customBanner is None: self.banner=customBanner
        super(QIPythonWidget, self).__init__(*args,**kwargs)
        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel_manager.kernel.gui = 'qt'
        self.kernel_client = kernel_client = self._kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
            guisupport.get_app_qt5().exit()            
        self.exit_requested.connect(stop)

    def pushVariables(self,variableDict):
        """ Given a dictionary containing name / value pairs, push those
        variables to the IPython console widget 
        """
        self.kernel_manager.kernel.shell.push(variableDict)
    def clearTerminal(self):
        """ Clears the terminal """
        self._control.clear()    
    def printText(self,text):
        """ Prints some plain text to the console """
        self._append_plain_text(text)        
    def executeCommand(self,command):
        """ Execute a command in the frame of the console widget """
        self._execute(command,False)


class IPythonWidget(QWidget):
    """ Main GUI Widget including an IPython Console widget inside vertical
    layout 
    """ 
    
    def __init__(self, parent=None):
        super(IPythonWidget, self).__init__(parent)
        layout = QVBoxLayout(self)
        ipyConsole = QIPythonWidget(customBanner="Welcome to Sesame IPython console\n")
        layout.addWidget(ipyConsole)        
        # This allows the variable foo and method print_process_id to be accessed from the ipython console
        # ipyConsole.pushVariables({"foo":43,"print_process_id":print_process_id})
        # ipyConsole.printText("The variable 'foo' and the method 'print_process_id()' are available. Use the 'whos' command for information.")                           

def print_process_id():
    print('Process ID is:', os.getpid())




class TableWidget(QWidget):
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
        self.viewBox = ViewBox()
        self.settingsBox = SettingsBox(self.viewBox)
        self.tab1Layout.addWidget(BuilderBox(self.settingsBox))
        self.tab1Layout.addWidget(self.settingsBox)
        self.tab1Layout.addWidget(self.viewBox)

        #============================================
        #  tab2: run the simulation
        #============================================
        self.tab2Layout = QHBoxLayout(self.tab2)
        self.simulation = Simulation(self.parent)
        self.tab2Layout.addWidget(self.simulation)

        #============================================
        #  tab3: analyze simulation results
        #============================================
        self.tab3Layout = QHBoxLayout(self.tab3)
        self.analysis = Analysis(self)
        self.tab3Layout.addWidget(self.analysis)
