import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
 
import numpy as np 

from .plotbox import *
from .common import parseSettings, slotError


class BuilderBox(QWidget):
    def __init__(self, parent=None):
        super(BuilderBox, self).__init__(parent)

        self.tabLayout = QHBoxLayout()
        self.setLayout(self.tabLayout)

        self.builder1()
        self.builder2() 
        self.builder3() 


    def builder1(self):
        layout = QVBoxLayout()
        self.tabLayout.addLayout(layout)

        #==============================
        # Grid settings
        #==============================
        gridBox = QGroupBox("Grid")
        gridBox.setMaximumWidth(400)
        gridLayout = QFormLayout()

        tip = QLabel("Each axis of the grid is a concatenation of sets of evenly spaced nodes. Edit the form with (x1, x2, number of nodes), (x2, x3, number of nodes),...")
        gridLayout.addRow(tip)
        tip.setStyleSheet("qproperty-alignment: AlignJustify;")
        tip.setWordWrap(True)

        self.g1 = QLineEdit()
        self.g2 = QLineEdit()
        self.g3 = QLineEdit()
        h1 = QHBoxLayout()
        h1.addWidget(self.g1)
        h1.addWidget(QLabel("m"))
        h2 = QHBoxLayout()
        h2.addWidget(self.g2)
        h2.addWidget(QLabel("m"))
        h3 = QHBoxLayout()
        h3.addWidget(self.g3)
        h3.addWidget(QLabel("m"))

        gridLayout.addRow("Grid x-axis", h1)
        gridLayout.addRow("Grid y-axis", h2)
        gridLayout.addRow("Grid z-axis", h3)

        gridBox.setLayout(gridLayout)
        layout.addWidget(gridBox)

        #==============================
        # Doping
        #==============================
        dopingBox = QGroupBox("Doping")
        dopingBox.setMaximumWidth(400)
        layoutD = QVBoxLayout()

        # Accceptor doping
        layout1 = QFormLayout()
        self.loc1 = QLineEdit("x > 1e-6")
        self.N1 = QLineEdit()
        layout1.addRow("Acceptor doping", QLabel())
        layout1.addRow("Location [m]", self.loc1)
        layout1.addRow("Concentration [m\u207B\u00B3]", self.N1)
        layoutD.addLayout(layout1)

        layoutD.addSpacing(40)

        # Donor doping
        layout2 = QFormLayout()
        self.loc2 = QLineEdit("x <= 1e-6")
        self.N2 = QLineEdit()
        layout2.addRow("Donor doping", QLabel())
        layout2.addRow("Location [m]", self.loc2)
        layout2.addRow("Concentration [m\u207B\u00B3]", self.N2)
        layoutD.addLayout(layout2)

        dopingBox.setLayout(layoutD)
        layout.addWidget(dopingBox)



    def builder2(self):
        matBox = QGroupBox("Materials")
        matBox.setMaximumWidth(400)
        vlayout = QVBoxLayout()
        matBox.setLayout(vlayout)
        self.tabLayout.addWidget(matBox)


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
        vlayout.addStretch()

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



    # display params of selected material
    def comboSelect(self):
        idx = self.box.currentIndex()
        mat = self.materials_list[idx]
        values = [mat[i] for i in self.rows]

        self.loc.setText(mat['location'])

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


    def builder3(self):
        layout3 = QVBoxLayout()
        self.tabLayout.addLayout(layout3)
       
        #=====================================================
        # Line and plane defects
        #=====================================================
        defectBox = QGroupBox("Defects")
        defectBox.setMaximumWidth(400)
        vlayout = QVBoxLayout()
        defectBox.setLayout(vlayout)
        layout3.addWidget(defectBox)

        # Add local charges
        self.hbox = QHBoxLayout()
        self.defectBox = QComboBox()
        self.defectBox.currentIndexChanged.connect(self.comboSelect2)
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
        self.cloc = QLineEdit("(x1, y1), (x2, y2)")
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

        self.rows2 = ("Energy", "Density", "sigma_e", "sigma_h", "Transition")
        self.ctable.setVerticalHeaderLabels(self.rows)

        columns = ("Value", "Unit")
        self.ctable.setHorizontalHeaderLabels(columns)

        self.defectValues = ["0.1", "1e13", "1e-15", "1e-15", "1/0"]
        self.units2 = ["eV", u"cm\u00B2", u"cm\u00B2", u"cm\u00B2", "NA"]

        for idx, (val, unit) in enumerate(zip(self.defectValues, self.units2)):
            self.ctable.setItem(idx,0, QTableWidgetItem(val))
            item = QTableWidgetItem(unit)
            item.setFlags(Qt.ItemIsEnabled)
            self.ctable.setItem(idx,1, item)

        
        #=====================================================
        # Generation
        #=====================================================
        genBox = QGroupBox("Generation rate")
        genBox.setMaximumWidth(400)
        genLayout = QVBoxLayout()
        genBox.setLayout(genLayout)
        layout3.addWidget(genBox)

        lbl = QLabel("Provide a number for uniform illumation, or a space-dependent function, or simply nothing for dark conditions. \nA single variable parameter is allowed and will be looped over during the simulation.")
        lbl.setStyleSheet("qproperty-alignment: AlignJustify;")
        lbl.setWordWrap(True)
        genLayout.addWidget(lbl)

        hlayout = QFormLayout()
        self.gen = QLineEdit("", self)
        hlayout.addRow("Expression", self.gen)
        self.paramName = QLineEdit("", self)
        hlayout.addRow("Paramater name", self.paramName)
        genLayout.addLayout(hlayout)

    # display params of selected defect
    def comboSelect2(self):
        idx = self.defectBox.currentIndex()
        defect = self.defects_list[idx]
        values = [defect[i] for i in self.rows2]

        self.cloc.setText(defect['location'])

        for idx, (val, unit) in enumerate(zip(values, self.units2)):
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
        values = [mt[i] for i in self.rows2]
        for idx, (val, unit) in enumerate(zip(values, self.units2)):
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
            key = self.rows2[row]
            try:
                self.defects_list[idx][key] = float(txt)
            except:
                self.defects_list[idx][key] = txt
    
    def getSystemSettings(self):
        settings = {}

        g1, g2, g3 = self.g1.text(), self.g2.text(), self.g3.text()
        if g1 != '' and g2 == '' and g3 == '':
            settings['grid'] = self.g1.text(),
        elif g1 != '' and g2 != '' and g3 == '':
            settings['grid'] = self.g1.text(), self.g2.text()
        elif g1 != '' and g2 != '' and g3 != '':
            settings['grid'] = self.g1.text(), self.g2.text(), self.g3.text()
        else:
            msg = QMessageBox()
            msg.setWindowTitle("Processing error")
            msg.setIcon(QMessageBox.Critical)
            msg.setText("The grid settings cannot be processed.")
            msg.setEscapeButton(QMessageBox.Ok)
            msg.exec_()
            return

        d = [{'location': self.loc1.text(), 'concentration': self.N1.text()},
             {'location': self.loc2.text(), 'concentration': self.N2.text()}]

        settings['doping'] = d
        settings['materials'] = self.materials_list
        settings['defects'] = self.defects_list
        settings['gen'] = self.gen.text(), self.paramName.text()
        return settings
