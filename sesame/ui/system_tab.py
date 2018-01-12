# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
 
import numpy as np 

from .plotbox import *


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
        h1.addWidget(QLabel("cm"))
        h2 = QHBoxLayout()
        h2.addWidget(self.g2)
        h2.addWidget(QLabel("cm"))
        h3 = QHBoxLayout()
        h3.addWidget(self.g3)
        h3.addWidget(QLabel("cm"))

        gridLayout.addRow("Grid x-axis", h1)
        gridLayout.addRow("Grid y-axis", h2)
        gridLayout.addRow("Grid z-axis", h3)

        gridBox.setLayout(gridLayout)
        layout.addWidget(gridBox)

        #==============================
        # Doping
        #==============================
        dopingBox = QGroupBox("Doping")
        dopingBox.setMinimumWidth(400)
        layoutD = QVBoxLayout()

        # Accceptor doping
        layout1 = QFormLayout()
        self.loc1 = QLineEdit("x > 1e-4")
        self.N1 = QLineEdit()
        layout1.addRow("Acceptor doping", QLabel())
        layout1.addRow("Location [cm]", self.loc1)
        layout1.addRow("Concentration [cm\u207B\u00B3]", self.N1)
        layoutD.addLayout(layout1)

        layoutD.addSpacing(10)

        # Donor doping
        layout2 = QFormLayout()
        self.loc2 = QLineEdit("x <= 1e-4")
        self.N2 = QLineEdit()
        layout2.addRow("Donor doping", QLabel())
        layout2.addRow("Location [cm]", self.loc2)
        layout2.addRow("Concentration [cm\u207B\u00B3]", self.N2)
        layoutD.addLayout(layout2)

        dopingBox.setLayout(layoutD)
        layout.addWidget(dopingBox)



    def builder2(self):
        matBox = QGroupBox("Materials")
        matBox.setMinimumWidth(400)
        vlayout = QVBoxLayout()
        matBox.setLayout(vlayout)
        self.tabLayout.addWidget(matBox)


        # Combo box to keep track of materials
        matLayout = QHBoxLayout()
        self.box = QComboBox()
        self.box.currentIndexChanged.connect(self.comboSelect)
        self.matNumber = -1

        # Add, remove and save buttons
        self.newButton = QPushButton("New")
        self.newButton.clicked.connect(self.addMat)
        self.newButton.setEnabled(False) # disable on start
        self.saveButton = QPushButton("Save")
        self.saveButton.clicked.connect(self.saveMat)
        self.removeButton = QPushButton("Remove")
        self.removeButton.setEnabled(False) # disabled on start
        self.removeButton.clicked.connect(self.removeMat)
        matLayout.addWidget(self.box)
        matLayout.addWidget(self.newButton)
        matLayout.addWidget(self.saveButton)
        matLayout.addWidget(self.removeButton)
        vlayout.addLayout(matLayout)

        # Reminder to save
        vlayout.addWidget(QLabel("Save a material before adding a new one."))

        # Location
        locLayout = QHBoxLayout()
        self.loc = QLineEdit("", self)
        self.lbl = QLabel("Location")
        locLayout.addWidget(self.lbl)
        locLayout.addWidget(self.loc)
        vlayout.addLayout(locLayout)

        # Label explaining how to write location
        self.ex = QLabel("Tip: Define the region for y < 1.5 µm or y > 2.5 µm with (y < 1.5e-6) | (y > 2.5e-6). Use the bitwise operators | for `or`, and & for `and`.")
        self.ex.setStyleSheet("qproperty-alignment: AlignJustify;")
        self.ex.setWordWrap(True)
        vlayout.addWidget(self.ex)


        # Table for material parameters
        self.table = QTableWidget()
        self.table.setRowCount(15)
        self.table.setColumnCount(2)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        vlayout.addWidget(self.table)

        # set table
        self.materials_list = []

        self.rows = ("Nc", "Nv", "Eg", "epsilon", "mass_e", "mass_h",\
                     "mu_e", "mu_h", "Et", "tau_e", "tau_h", "affinity",\
                     "B", "Cn", "Cp")
        columns = ("Value", "Unit")
        self.table.setVerticalHeaderLabels(self.rows)
        self.table.setHorizontalHeaderLabels(columns)

        self.units = [u"cm\u207B\u00B3", u"cm\u207B\u00B3", "eV", "NA", "NA", "NA",
                 u"cm\u00B2/(V s)",\
                 u"cm\u00B2/(V s)", "eV", "s", "s", "eV", u"cm\u00B3/s",\
                 u"cm\u2076/s", u"cm\u2076/s"]

        # Initialize table
        mt = {'Nc': 1e19, 'Nv': 1e19, 'Eg': 1, 'epsilon': 10, 'mass_e': 1, \
              'mass_h': 1, 'mu_e': 100, 'mu_h': 100, 'Et': 0, \
              'tau_e': 1e-6, 'tau_h': 1e-6, 'affinity': 0, \
              'B': 0, 'Cn': 0, 'Cp': 0, 'location': None}

        values = [mt[i] for i in self.rows]
        for idx, (val, unit) in enumerate(zip(values, self.units)):
            self.table.setItem(idx, 0, QTableWidgetItem(str(val)))
            item = QTableWidgetItem(unit)
            item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(idx,1, item)

        
        self.matNumber += 1
        idx = self.matNumber
        self.materials_list.append({})

        loc = self.loc.text()
        self.materials_list[idx]['location'] = loc

        # get params
        for row in range(15):
            item = self.table.item(row, 0)
            txt = item.text()
            key = self.rows[row]
            self.materials_list[idx][key] = float(txt)

        self.box.addItem("Material " + str(self.matNumber + 1))
        self.box.setCurrentIndex(self.matNumber)


    # display params of selected material
    def comboSelect(self):
        idx = self.box.currentIndex()
        self.matNumber = idx
        mat = self.materials_list[idx]
        values = [mat[i] for i in self.rows]

        self.loc.setText(mat['location'])

        for idx, (val, unit) in enumerate(zip(values, self.units)):
            self.table.setItem(idx,0, QTableWidgetItem(str(val)))

    # add new material 
    def addMat(self):
        mt = {'Nc': 1e19, 'Nv': 1e19, 'Eg': 1, 'epsilon': 10, 'mass_e': 1,\
                    'mass_h': 1, 'mu_e': 100, 'mu_h': 100, 'Et': 0,\
                    'tau_e': 1e-6, 'tau_h': 1e-6, 'affinity': 0,\
                    'B': 0, 'Cn': 0, 'Cp': 0, 'location': None}

        # 1. reinitialize location
        self.loc.clear()

        # 2. reinitialize table
        values = [mt[i] for i in self.rows]
        for idx, (val, unit) in enumerate(zip(values, self.units)):
            self.table.setItem(idx,0, QTableWidgetItem(str(val)))
        self.table.show()

        self.matNumber = self.materials_list.__len__()
        idx = self.matNumber
        self.materials_list.append({})

        # 3. save standard quantities for the material
        # get location
        loc = self.loc.text()
        self.materials_list[idx]['location'] = loc
        # get params
        for row in range(15):
            item = self.table.item(row, 0)
            txt = item.text()
            key = self.rows[row]
            self.materials_list[idx][key] = float(txt)

        # 4. Increment material number in combo box
        self.box.addItem("Material " + str(self.matNumber + 1))
        self.box.setCurrentIndex(self.matNumber)

        # 5. Disable remove and new buttons
        self.removeButton.setEnabled(False)
        self.newButton.setEnabled(False)
        self.saveButton.setEnabled(True)

    # store data entered
    def saveMat(self):
        # set ID of material
        idx = self.matNumber

        # get location
        loc = self.loc.text()
        self.materials_list[idx]['location'] = loc

        # get params
        for row in range(15):
            item = self.table.item(row, 0)
            txt = item.text()
            key = self.rows[row]
            self.materials_list[idx][key] = float(txt)

        # disable save, enable remove and new buttons
        self.newButton.setEnabled(True)
        if len(self.materials_list) > 1:
            self.removeButton.setEnabled(True)

    # remove a material
    def removeMat(self):
        if len(self.materials_list) > 1:
            # remove from list
            idx = self.box.currentIndex()
            del self.materials_list[idx]
            self.matNumber -= 1
            # remove from combo box
            self.box.removeItem(idx)
            # rename all the others
            for idx in range(self.box.count()):
                self.box.setItemText(idx, "Material " + str(idx + 1))

        # disable remove if nothing to remove, enable new mat
        if len(self.materials_list) == 1:
            self.removeButton.setEnabled(False)
            self.newButton.setEnabled(True)
            self.saveButton.setEnabled(False)

    def builder3(self):
        layout3 = QVBoxLayout()
        self.tabLayout.addLayout(layout3)
       
        #=====================================================
        # Line and plane defects
        #=====================================================
        defectBox = QGroupBox("Planar Defects")
        defectBox.setMinimumWidth(400)
        vlayout = QVBoxLayout()
        defectBox.setLayout(vlayout)
        layout3.addWidget(defectBox)

        # Combo box to keep track of defects
        self.hbox = QHBoxLayout()
        self.defectBox = QComboBox()
        self.hbox.addWidget(self.defectBox)
        self.defectBox.currentIndexChanged.connect(self.comboSelect2)
        self.defectNumber = -1

        # Add and save buttons
        self.defectButton = QPushButton("New")
        self.defectButton.clicked.connect(self.addDefects)
        self.saveButton2 = QPushButton("Save")
        self.saveButton2.clicked.connect(self.saveDefect)
        self.saveButton2.setEnabled(False) # disabled on start
        self.removeButton2 = QPushButton("Remove")
        self.removeButton2.setEnabled(False) # disabled on start
        self.removeButton2.clicked.connect(self.removeDefect)
        self.hbox.addWidget(self.defectButton)
        self.hbox.addWidget(self.saveButton2)
        self.hbox.addWidget(self.removeButton2)

        vlayout.addLayout(self.hbox)

        # Reminder to save
        vlayout.addWidget(QLabel("Save a defect before adding a new one."))

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
        self.ctable.setVerticalHeaderLabels(self.rows2)

        columns = ("Value", "Unit")
        self.ctable.setHorizontalHeaderLabels(columns)

        self.defectValues = ["0.1", "1e13", "1e-15", "1e-15", "1/0"]
        self.units2 = ["eV", u"cm\u207B\u00B2", u"cm\u00B2", u"cm\u00B2", "NA"]

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
        hlayout.addRow("Expression [cm\u207B\u00B3s\u207B\u00B9]", self.gen)
        self.paramName = QLineEdit("", self)
        hlayout.addRow("Paramater name", self.paramName)
        genLayout.addLayout(hlayout)

    # display params of selected defect
    def comboSelect2(self):
        idx = self.defectBox.currentIndex()
        self.defectNumber = idx
        if len(self.defects_list) == 0:
            # nothing to display so, hide the table
            self.cloc.hide()
            self.ctable.hide()
        else:
            # display settings of desired defect
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

        # reinitialize location
        self.cloc.clear()
        self.cloc.insert("(x1, y1), (x2, y2)")
        self.cloc.show()
        self.clbl.show()

        # reinitialize table
        values = [mt[i] for i in self.rows2]
        for idx, (val, unit) in enumerate(zip(values, self.units2)):
            self.ctable.setItem(idx,0, QTableWidgetItem(str(val)))
        self.ctable.show()

        self.defectNumber = self.defects_list.__len__()
        idx = self.defectNumber
        self.defects_list.append({})

        # get location
        loc = self.cloc.text()
        self.defects_list[idx]['location'] = loc

        # get params
        for row in range(5):
            item = self.ctable.item(row, 0)
            txt = item.text()
            key = self.rows2[row]
            try:
                self.defects_list[idx][key] = float(txt)
            except:
                self.defects_list[idx][key] = txt

        # add "defect number" to combo box
        self.defectBox.addItem("Defect " + str(self.defectNumber + 1))
        self.defectBox.setCurrentIndex(self.defectNumber)

        # Disable remove and new buttons, enable save button
        #self.removeButton2.setEnabled(False)
        #self.defectButton.setEnabled(False)
        #self.saveButton2.setEnabled(True)


    # store data entered
    def saveDefect(self):
        # set ID of defect
        idx = self.defectNumber

        # get location
        loc = self.cloc.text()
        self.defects_list[idx]['location'] = loc

        # get params
        for row in range(5):
            item = self.ctable.item(row, 0)
            txt = item.text()
            key = self.rows2[row]
            try:
                self.defects_list[idx][key] = float(txt)
            except:
                self.defects_list[idx][key] = txt

        # enable remove and new buttons
        #self.defectButton.setEnabled(True)
        #self.removeButton2.setEnabled(True)

    # remove a defect
    def removeDefect(self):
        if len(self.defects_list) > 0:
            # remove from list
            idx = self.defectBox.currentIndex()
            del self.defects_list[idx]
            self.defectNumber -= 1
            # remove from combo box
            self.defectBox.removeItem(idx)
            # rename all the others
            for idx in range(self.defectBox.count()):
                self.defectBox.setItemText(idx, "Defect " + str(idx + 1))

        # disable remove if nothing to remove, enable new defect
        self.defectButton.setEnabled(True)
        if len(self.defects_list) > 0:
            self.removeButton2.setEnabled(False)
        if len(self.defects_list) == 0:
            self.removeButton2.setEnabled(False)
            self.saveButton2.setEnabled(False)
            self.clbl.hide()

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


