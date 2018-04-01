#!/usr/bin/env python3
# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

from sesame.ui import mainwindow
from PyQt5.QtWidgets import QApplication, QMessageBox, qApp
from PyQt5 import QtGui, QtCore
import sys
import ctypes
import traceback


def exception_handler(type_, value, traceback_):
    if qApp.thread() is QtCore.QThread.currentThread():
        p = traceback.format_exc()
        msg = QMessageBox()
        msg.setWindowTitle("Sesame error")
        msg.setIcon(QMessageBox.Critical)
        msg.setText("An unhandled error occurred. Check the log below for more information")
        msg.setDetailedText(p)
        msg.setEscapeButton(QMessageBox.Ok)
        msg.exec_()


if sys.platform == 'win32':
    appID = 'sesameGUI'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appID)

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

sys.excepthook = exception_handler

sys.exit(app.exec_())
