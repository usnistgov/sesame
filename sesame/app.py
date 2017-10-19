#!/usr/bin/env python3

from sesame.ui import mainwindow
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
window = mainwindow.Window()

sys.exit(app.exec_())
