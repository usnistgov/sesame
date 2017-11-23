#!/usr/bin/env python3

from sesame.ui import mainwindow
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtGui
import sys

app = QApplication(sys.argv)

stylesheet = """
QGroupBox { 
    border: 1px solid gray;
    border-radius: 9px;
    margin-top: 0.5em;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 3px 0 3px;
}
"""
app.setStyleSheet(stylesheet)


window = mainwindow.Window()

sys.exit(app.exec_())