from .. import Builder
from ast import literal_eval as ev
import numpy as np
import traceback
import types
from functools import wraps
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMessageBox
import logging


def slotError(*args):
    if len(args) == 0 or isinstance(args[0], types.FunctionType):
        args = []
    @QtCore.pyqtSlot(*args)
    def slotdecorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                func(*args)
            except:
                p = traceback.format_exc()
                # Dialog box
                msg = QMessageBox()
                msg.setWindowTitle("Processing error")
                msg.setIcon(QMessageBox.Critical)
                msg.setText("An error occurred when processing your settings.")
                msg.setDetailedText(p)
                msg.setEscapeButton(QMessageBox.Ok)
                msg.exec_()
        return wrapper

    return slotdecorator


def parseGrid(grid):
    # find the number of regions to concatenate
    regions = 1
    for i in grid:
        if isinstance(i, tuple):
            regions += 1

    # parse a one-dimensional grid (either x or y or z)
    if regions == 1: # only one linspace
        x = np.linspace(grid[0], grid[1], grid[2])
    else: # several regions were defined
        x = np.linspace(grid[0][0], grid[0][1], grid[0][2], endpoint=False)
        for item in grid[1:-1]:
            xx = np.linspace(item[0], item[1], item[2], endpoint=False)
            x = np.concatenate((x, xx))
        xx = np.linspace(grid[-1][0], grid[-1][1], grid[-1][2])
        x = np.concatenate((x, xx))
    return x

def parseLocation(location, dimension):
    # return a lambda function defining a region
    # 0. check if string is empty
    if location == "": # a single material
        function = lambda pos: True
    else:
        # 1. replace x, y, z by pos or pos[0], pos[1], pos[2] in the string
        if dimension == 1:
            location = location.replace("x", "pos")
        elif dimension == 2:
            location = location.replace("x", "pos[0]")
            location = location.replace("y", "pos[1]")
        elif dimension == 3:
            location = location.replace("x", "pos[0]")
            location = location.replace("y", "pos[1]")
            location = location.replace("z", "pos[2]")
        # 2. define function
        function = lambda pos: eval(location)
    return function


def parseSettings(settings):
    # 1. create grid
    grid = settings['grid']
    dimension = len(grid)

    xgrid = ev(grid[0])
    xpts = parseGrid(xgrid)
    ypts = None
    zpts = None
    if dimension == 2 or dimension == 3:
        ygrid = ev(grid[1])
        ypts = parseGrid(ygrid)
    if dimension == 3:
        zgrid = ev(grid[2])
        zpts = parseGrid(zgrid)
    # build a sesame system
    system = Builder(xpts, ypts, zpts)
        
    # 2. set contacts boundary conditions
    contacts = settings['contacts']
    Sn_left, Sp_left, Sn_right, Sp_right = [float(i) for i in contacts]
    system.contacts(Sn_left, Sp_left, Sn_right, Sp_right)

    # 3. set materials
    materials = settings['materials']
    for mat in materials:
        location = mat['location']
        # define a function that returns true/false dpending on location
        f = parseLocation(location, dimension)
        system.add_material(mat, f)

    # 4. set the doping
    doping = settings['doping']
    acceptor, donor = doping

    # location function, and concentration
    f = parseLocation(acceptor['location'], dimension)
    N = float(acceptor['concentration'])
    system.add_acceptor(N, f)

    # location function, and concentration
    f = parseLocation(donor['location'], dimension)
    N = float(donor['concentration'])
    system.add_donor(N, f)

    # 5. set the defects if present
    defects = settings['defects']
    defects = defects
    for defect in defects:
        loc = ev(defect['location'])
        N = float(defect['Density'])
        E = float(defect['Energy'])
        sh = float(defect['sigma_h'])
        se = float(defect['sigma_e'])
        transition = defect['Transition'].replace("/", ",")
        transition = (ev(transition))

        if len(loc) == 2:
            system.add_line_defects(loc, N, se, sigma_h=sh, E=E,\
                                    transition=transition)
        elif len(loc) == 4:
            system.add_plane_defects(loc, N, se, sigma_h=sh, E=E,\
                                     transition=transition)
    return system

