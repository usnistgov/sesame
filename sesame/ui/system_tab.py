import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
 
import numpy as np 

from .plotbox import *
from .common import parseSettings, slotError


class BuilderBox(QWidget):
    def __init__(self, parent):
        super(BuilderBox, self).__init__(parent)

        # define widget
        self.setMaximumWidth(200)
        self.setMinimumWidth(200)
        layout = QVBoxLayout()

        self.formGroupBox = QGroupBox("Builder")
 
        self.list = QListWidget()
        self.list.insertItem (0, 'Grid' )
        self.list.insertItem (1, 'Contacts' )
        self.list.insertItem (2, 'Materials' )
        self.list.insertItem (3, 'Doping' )
        self.list.insertItem (4, 'Defects' )
        self.list.insertItem (5, 'Generation' )
        self.list.insertItem (6, 'Plot system' )
        layout.addWidget(self.list)
        self.list.currentRowChanged.connect(self.display)

        layout.setContentsMargins(0, 0, 0, 0)
        self.formGroupBox.setLayout(layout)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)

        self.setLayout(mainLayout)

        self.stack = parent.Stack

    def display(self, i):
        self.stack.setCurrentIndex(i+1)
    

class SettingsBox(QWidget):
    def __init__(self, parent):
        super(SettingsBox, self).__init__(parent)

        self.parentViewBox = parent

        self.setMinimumWidth(500)
        self.layout = QVBoxLayout()

        self.formGroupBox = QGroupBox("Settings")

        # empty page
        self.stack0 = QWidget()
        layout = QHBoxLayout()
        self.stack0.setLayout(layout)

        # create stacked widget
        self.Stack = QStackedWidget(self)
        self.Stack.addWidget (self.stack0)

        self.layout.addWidget(self.Stack)

        self.formGroupBox.setLayout(self.layout)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)

        self.setLayout(mainLayout)

    def make_stack(self, dimension):
        self.grid = SettingsGrid(self, dimension)
        self.contacts = SettingsContacts(self)
        self.materials = SettingsMaterials(self)
        self.doping = SettingsDoping(self)
        self.defects = SettingsDefects(self)
        self.gen = SettingsGen(self)
        self.plot = SettingsPlot(self, self.parentViewBox)

    def get_settings(self):
        settings = {}
        settings['grid'] = self.grid.get_data()
        settings['contacts'] = self.contacts.get_data()
        settings['materials'] = self.materials.get_data()
        settings['doping'] = self.doping.get_data()
        settings['defects'] = self.defects.get_data()
        settings['gen'] = self.gen.get_data()
        return settings
        
        
class SettingsGrid(QWidget):
    def __init__(self, parent, dimension):
        super(SettingsGrid, self).__init__(parent)

        self.dimension = dimension
        self.stackUI(parent)

    def stackUI(self, parent):
        layout = QFormLayout()
        self.g1 = QLineEdit("(x1, x2, number of nodes), (x2, x3, number of nodes), ...")
        self.g2 = QLineEdit("(y1, y2, number of nodes), (y2, y3, number of nodes), ...")
        self.g3 = QLineEdit("(z1, z2, number of nodes), (z2, z3, number of nodes), ...")

        if self.dimension == 1:
            layout.addRow("Grid x-axis [m]", self.g1)
        elif self.dimension == 2:
            layout.addRow("Grid x-axis [m]", self.g1)
            layout.addRow("Grid y-axis [m]", self.g2)
        elif self.dimension == 3:
            layout.addRow("Grid x-axis [m]", self.g1)
            layout.addRow("Grid y-axis [m]", self.g2)
            layout.addRow("Grid z-axis [m]", self.g3)

        stack = QWidget()
        stack.setLayout(layout)
        parent.Stack.addWidget(stack)

    def get_data(self):
        if self.dimension == 1:
            return self.g1.text(),
        elif self.dimension == 2:
            return self.g1.text(), self.g2.text()
        elif self.dimension == 3:
            return self.g1.text(), self.g2.text(), self.g3.text()

class SettingsContacts(QWidget):
    def __init__(self, parent):
        super(SettingsContacts, self).__init__(parent)

        self.stackUI(parent)

    def stackUI(self, parent):
        layout = QFormLayout()
        self.g4 = QLineEdit("", self)
        self.g5 = QLineEdit("", self)
        self.g6 = QLineEdit("", self)
        self.g7 = QLineEdit("", self)
        layout.addRow("Electron surface recombination velocity in x=0 [m/s]", self.g4)
        layout.addRow("Hole surface recombination velocity in x=0 [m/s]", self.g5)
        layout.addRow("Electron surface recombination velocity in x=L [m/s]", self.g6)
        layout.addRow("Hole surface recombination velocity in x=L [m/s]", self.g7)

        stack = QWidget()
        stack.setLayout(layout)
        parent.Stack.addWidget(stack)

    def get_data(self):
        return [self.g4.text(), self.g5.text(), self.g6.text(), self.g7.text()]
        

class SettingsGen(QWidget):
    def __init__(self, parent):
        super(SettingsGen, self).__init__(parent)

        self.stackUI(parent)

    def stackUI(self, parent):
        layout = QVBoxLayout()
        lbl = QLabel("Provide a number for uniform illumation, or a space-dependent function, or simply nothing for dark conditions. \nA single variable parameter is allowed and will be looped over during the simulation.")
        lbl.setStyleSheet("qproperty-alignment: AlignJustify;")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        hlayout = QFormLayout()
        self.gen = QLineEdit("", self)
        hlayout.addRow("Expression", self.gen)
        self.paramName = QLineEdit("", self)
        hlayout.addRow("Paramater name", self.paramName)

        layout.addLayout(hlayout)
        layout.addStretch()

        stack = QWidget()
        stack.setLayout(layout)
        parent.Stack.addWidget(stack)

    def get_data(self):
        return self.gen.text(), self.paramName.text()

class SettingsMaterials(QWidget):
    def __init__(self, parent):
        super(SettingsMaterials, self).__init__(parent)

        self.stackUI(parent)

    def stackUI(self, parent):
        vlayout = QVBoxLayout()

        # Combo box to keep track of materials
        matLayout = QHBoxLayout()
        self.box = QComboBox()
        self.box.currentIndexChanged.connect(self.comboSelect)
        self.matNumber = -1

        # Add and save buttons
        button = QPushButton("Add material")
        button.clicked.connect(self.addMat)
        saveButton = QPushButton("Save")
        saveButton.clicked.connect(self.saveMat)
        matLayout.addWidget(self.box)
        matLayout.addWidget(button)
        matLayout.addWidget(saveButton)
        vlayout.addLayout(matLayout)

        # Reminder to save
        vlayout.addWidget(QLabel("Save a material before adding a new one."))
        # vlayout.addStretch()

        # Location
        locLayout = QHBoxLayout()
        self.loc = QLineEdit("", self)
        self.lbl = QLabel("Location")
        locLayout.addWidget(self.lbl)
        locLayout.addWidget(self.loc)
        self.loc.hide()
        self.lbl.hide()
        vlayout.addLayout(locLayout)

        # Label explaining how to write location
        self.ex = QLabel("Tip: Define the region for y < 1.5 µm or y > 2.5 µm with \n(y < 1.5e-6) | (y > 2.5e-6) \nUse the bitwise operators | for `or`, and & for `and`.")
        self.ex.setStyleSheet("qproperty-alignment: AlignJustify;")
        self.ex.setWordWrap(True)
        self.ex.hide()
        vlayout.addWidget(self.ex)


        # Table for material parameters
        self.table = QTableWidget()
        self.table.setRowCount(15)
        self.table.setColumnCount(2)
        self.table.hide()
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        vlayout.addWidget(self.table)
        vlayout.addStretch()

        # set table
        self.materials_list = []

        self.rows = ("Nc", "Nv", "Eg", "epsilon", "mass_e", "mass_h",\
                     "mu_e", "mu_h", "Et", "tau_e", "tau_h", "band_offset",\
                     "B", "Cn", "Cp")
        columns = ("Value", "Unit")
        self.table.setVerticalHeaderLabels(self.rows)
        self.table.setHorizontalHeaderLabels(columns)

        self.units = [u"m\u207B\u00B3", u"m\u207B\u00B3", "eV", "NA", "NA", "NA",
                 u"m\u00B2/(V s)",\
                 u"m\u00B2/(V s)", "eV", "s", "s", "eV", u"m\u00B3/s",\
                 u"m\u2076/s", u"m\u2076/s"]

        for idx, unit in enumerate(self.units):
            item = QTableWidgetItem(unit)
            item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(idx,1, item)


        stack = QWidget()
        stack.setLayout(vlayout)
        parent.Stack.addWidget(stack)

    # returns the materials list outside the class
    def get_data(self):
        return self.materials_list

    # display params of selected material
    def comboSelect(self):
        idx = self.box.currentIndex()
        mat = self.materials_list[idx]
        values = [mat[i] for i in self.rows]

        for idx, (val, unit) in enumerate(zip(values, self.units)):
            self.table.setItem(idx,0, QTableWidgetItem(str(val)))

    # add new material 
    def addMat(self):
        mt = {'Nc': 1e25, 'Nv': 1e25, 'Eg': 1, 'epsilon': 1, 'mass_e': 1,\
                    'mass_h': 1, 'mu_e': 100e-4, 'mu_h': 100e-4, 'Et': 0,\
                    'tau_e': 1e-6, 'tau_h': 1e-6, 'band_offset': 0,\
                    'B': 0, 'Cn': 0, 'Cp': 0, 'location': None}
        self.materials_list.append(mt)

        # 1. reinitialize location
        self.loc.clear()
        self.loc.show()
        self.lbl.show()
        self.ex.show()

        # 2. reinitialize table
        values = [mt[i] for i in self.rows]
        for idx, (val, unit) in enumerate(zip(values, self.units)):
            self.table.setItem(idx,0, QTableWidgetItem(str(val)))
        self.table.show()


    # store data entered
    def saveMat(self):
        # set combo box material ID
        self.matNumber += 1
        self.box.addItem("Material " + str(self.matNumber+1))
        self.box.setCurrentIndex(self.matNumber)
        # get ID of material
        idx = self.box.currentIndex()

        # get location
        loc = self.loc.text()
        self.materials_list[idx]['location'] = loc

        # get params
        for row in range(15):
            item = self.table.item(row, 0)
            txt = item.text()
            key = self.rows[row]
            self.materials_list[idx][key] = float(txt)

class SettingsDoping(QWidget):
    def __init__(self, parent):
        super(SettingsDoping, self).__init__(parent)

        layout = QVBoxLayout()

        # Accceptor doping
        layout1 = QFormLayout()
        self.loc1 = QLineEdit("x > 1e-6")
        self.N1 = QLineEdit()
        layout1.addRow("Acceptor doping", QLabel())
        layout1.addRow("Location [m]", self.loc1)
        layout1.addRow("Concentration [m\u207B\u00B3]", self.N1)
        layout.addLayout(layout1)

        layout.addSpacing(40)

        # Donor doping
        layout2 = QFormLayout()
        self.loc2 = QLineEdit("x <= 1e-6")
        self.N2 = QLineEdit()
        layout2.addRow("Donor doping", QLabel())
        layout2.addRow("Location [m]", self.loc2)
        layout2.addRow("Concentration [m\u207B\u00B3]", self.N2)
        layout.addLayout(layout2)

        layout.addStretch()

        stack = QWidget()
        stack.setLayout(layout)
        parent.Stack.addWidget(stack)


    def get_data(self):
        d = [{'location': self.loc1.text(), 'concentration': self.N1.text()},
             {'location': self.loc2.text(), 'concentration': self.N2.text()}]
        return d

class SettingsDefects(QWidget):
    def __init__(self, parent):
        super(SettingsDefects, self).__init__(parent)

        self.stackUI(parent)

    def stackUI(self, parent):
        vlayout = QVBoxLayout()

        # Add local charges
        self.hbox = QHBoxLayout()
        self.defectBox = QComboBox()
        self.defectBox.currentIndexChanged.connect(self.comboSelect)
        self.defectButton = QPushButton("Add defects")
        saveButton = QPushButton("Save")
        saveButton.clicked.connect(self.saveDefect)
        self.hbox.addWidget(self.defectBox)
        self.hbox.addWidget(self.defectButton)
        self.hbox.addWidget(saveButton)

        self.defectNumber = -1
        self.defectButton.clicked.connect(self.addDefects)
        vlayout.addLayout(self.hbox)

        # Reminder to save
        vlayout.addWidget(QLabel("Save a defect before adding a new one."))
        vlayout.addStretch()

        self.clocLayout = QHBoxLayout()
        self.cloc = QLineEdit("(x1, y1), (x2, y2)", self)
        self.clbl = QLabel("Location")
        self.clocLayout.addWidget(self.clbl)
        self.clocLayout.addWidget(self.cloc)
        self.cloc.hide()
        self.clbl.hide()
        vlayout.addLayout(self.clocLayout)

        # Table for defect properties
        self.ctable = QTableWidget()
        self.ctable.setRowCount(5)
        self.ctable.setColumnCount(2)
        self.ctable.hide()
        cheader = self.ctable.horizontalHeader()
        cheader.setStretchLastSection(True)
        vlayout.addWidget(self.ctable)
        vlayout.addStretch()


        # set table
        self.defects_list = []

        self.rows = ("Energy", "Density", "sigma_e", "sigma_h", "Transition")
        self.ctable.setVerticalHeaderLabels(self.rows)

        columns = ("Value", "Unit")
        self.ctable.setHorizontalHeaderLabels(columns)

        self.defectValues = ["0.1", "1e13", "1e-15", "1e-15", "1/0"]
        self.units = ["eV", u"cm\u00B2", u"cm\u00B2", u"cm\u00B2", "NA"]

        for idx, (val, unit) in enumerate(zip(self.defectValues, self.units)):
            self.ctable.setItem(idx,0, QTableWidgetItem(val))
            item = QTableWidgetItem(unit)
            item.setFlags(Qt.ItemIsEnabled)
            self.ctable.setItem(idx,1, item)

        stack = QWidget()
        stack.setLayout(vlayout)
        parent.Stack.addWidget(stack)

    # returns the defects list outside the class
    def get_data(self):
        return self.defects_list

    # display params of selected defect
    def comboSelect(self):
        idx = self.defectBox.currentIndex()
        mat = self.defects_list[idx]
        values = [mat[i] for i in self.rows]

        for idx, (val, unit) in enumerate(zip(values, self.units)):
            self.ctable.setItem(idx,0, QTableWidgetItem(str(val)))
            item = QTableWidgetItem(unit)


    # add new defect
    def addDefects(self):
        mt = {'Energy': "0", 'Density': "1e13", 'sigma_e':
        "1e-15", 'sigma_h': "1e-15", 'Transition': "1/0", 'location': None}
        self.defects_list.append(mt)

        # 2. reinitialize location
        self.cloc.clear()
        self.cloc.insert("(x1, y1), (x2, y2)")
        self.cloc.show()
        self.clbl.show()

        # 3. reinitialize table
        values = [mt[i] for i in self.rows]
        for idx, (val, unit) in enumerate(zip(values, self.units)):
            self.ctable.setItem(idx,0, QTableWidgetItem(str(val)))
        self.ctable.show()


    # store data entered
    def saveDefect(self):
        # add "material number" to combo box
        self.defectNumber += 1
        self.defectBox.addItem("Defect " + str(self.defectNumber+1))
        self.defectBox.setCurrentIndex(self.defectNumber)
        # get ID of defect
        idx = self.defectBox.currentIndex()

        # get location
        loc = self.cloc.text()
        self.defects_list[idx]['location'] = loc

        # get params
        for row in range(4):
            item = self.ctable.item(row, 0)
            txt = item.text()
            key = self.rows[row]
            try:
                self.defects_list[idx][key] = float(txt)
            except:
                self.defects_list[idx][key] = txt
                
class SettingsPlot(QWidget):
    def __init__(self, parent, parentViewBox):
        super(SettingsPlot, self).__init__(parent)

        self.stackUI(parent)
        self.parent = parent
        self.parentViewBox = parentViewBox

    def stackUI(self, parent):
        layout = QVBoxLayout()
        lbl = QLabel("Choose a property of the system to visualize in the plotting area on the right.")
        lbl.setStyleSheet("qproperty-alignment: AlignJustify;")
        lbl.setWordWrap(True)

        self.propertyBox = QComboBox()
        self.propertyBox.addItem("Choose one")
        self.propertyBox.addItem("Lines defects")
        self.propertyBox.addItem("Planes defects")
        self.propertyBox.addItem("Electron mobility")
        self.propertyBox.addItem("Hole mobility")
        self.propertyBox.addItem("Electron lifetime")
        self.propertyBox.addItem("Hole lifetime")

        self.plotButton = QPushButton("Plot")
        self.plotButton.clicked.connect(self.displayPlot)

        layout.addWidget(lbl)
        layout.addWidget(self.propertyBox)
        layout.addWidget(self.plotButton)

        layout.addStretch()

        stack = QWidget()
        stack.setLayout(layout)
        parent.Stack.addWidget(stack)
    
    @slotError("bool")
    def displayPlot(self, checked):
        prop = self.propertyBox.currentText()
        settings = self.parent.get_settings()
        system = parseSettings(settings)
        if prop == "Electron mobility":
            data = system.mu_e
        elif prop == "Hole mobility":
            data = system.mu_h
        elif prop == "Electron lifetime":
            data = system.tau_e
        elif prop == "Hole lifetime":
            data = system.tau_h
        elif prop == "Lines defects":
            data = 'line'
        elif prop == "Planes defects":
            data = 'plane'

        self.parentViewBox.plotData(system, data)

 
class ViewBox(QWidget):
    def __init__(self):
        super(ViewBox, self).__init__()

        self.setMinimumWidth(400)
        self.layout = QVBoxLayout()

        self.formGroupBox = QGroupBox("View")
 
        self.mpl = MplWindow()
        self.layout.addWidget(self.mpl)
 
        self.formGroupBox.setLayout(self.layout)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)
        self.setLayout(mainLayout)

    def plotData(self, system, data):
        self.mpl.figure.clear()
        if system and isinstance(data, np.ndarray):
            plotter.plot(system, data, fig=self.mpl.figure)

        elif data == 'line':
            plotter.plot_line_defects(system, fig=self.mpl.figure)

        elif data == 'plane':
            plotter.plot_plane_defects(system, fig=self.mpl.figure)

        self.mpl.canvas.draw()
