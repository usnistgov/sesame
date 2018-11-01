# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

from .. import Builder
from .. utils import isfloat
from ast import literal_eval as ev
from scipy.interpolate import interp1d
import numpy as np
from .onesun_data import *
import traceback
import types
from functools import wraps
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMessageBox
import os.path

import matplotlib.pyplot as plt


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
        # 1. replace x, y by pos or pos[0], pos[1] in the string
        if dimension == 1:
            location = location.replace("x", "pos")
        elif dimension == 2:
            location = location.replace("x", "pos[0]")
            location = location.replace("y", "pos[1]")
        # 2. define function
        function = lambda pos: eval(location)
    return function


def parseAlphaFile(file):

    _lambda = []
    _alpha = []
    # assumed units of file:  lambda in [nm], absorption in [1/m]
    if os.path.isfile(file) is False:
        msg = QMessageBox()
        msg.setWindowTitle("Processing error")
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Absorption file not found")
        msg.setEscapeButton(QMessageBox.Ok)
        msg.exec_()
        return

    absfile = open(file,"r")
    while True:
        line = absfile.readline()
        # end of file reached - break
        if not line:
            break
        data = line.split()
        if len(data) == 2:
            # if line consists of two floats, read into lambda and alpha matrices
            if isfloat(data[0]) and isfloat(data[1]):
                _lambda.append(float(data[0]))
                _alpha.append(float(data[1]))

    _lambda = np.asarray(_lambda)
    _alpha = np.asarray(_alpha)
    _alpha = _alpha * 1e-2 # convert to 1/cm
    return _lambda, _alpha


def getgeneration(lambda_power, power, lambda_alpha, alpha, xpts):

    gen = []
    if alpha.size == 1:
        lambda_alpha = np.linspace(300,3000,745)
        if type(alpha) is np.ndarray:
            alpha = np.asarray(alpha, dtype='float64')
            alpha = alpha * np.ones(745)
        else:
            alpha = float(alpha) * np.ones(745)

    if power.size==0 or alpha.size==0:
        gen = np.zeros(xpts.size)
        return gen

    #### interpolate absorption spectrum according to power spectrum
    ind1 = np.abs(lambda_power - lambda_alpha[0]).argmin()
    ind2 = np.abs(lambda_power - lambda_alpha[-1]).argmin()

    f = interp1d(lambda_alpha, alpha, kind='cubic')
    alpha = f(lambda_power[ind1:ind2])
    power = power[ind1:ind2]
    llambda = lambda_power[ind1:ind2]

    ####################################################

    hc = 6.62607004e-34 * 299792458  # units: [J m]
    gen = np.zeros(xpts.size)
    nP = power.size
    # integrate over power/absorption spectra to get generation profile
    # assume power is given in W/cm^2
    for c in range(0, nP):
        if alpha[c] < 0:
            continue
        if c == 0:
            dl = .5*(llambda[c+1])
        elif c == nP-1:
            dl = .5*(llambda[-1] - llambda[-2])
        else:
            dl = .5*(llambda[c+1] - llambda[c-1])
        # convert power to flux:  P = h*c/lambda * flux
        flux = power[c] * llambda[c]*1e-9 / (hc)  # 1/(cm^2 * sec)
        gen = gen + alpha[c] * flux * np.exp(-alpha[c]*xpts) * dl

    return gen


def parseSettings(settings):
    # 1. create grid
    grid = settings['grid']
    dimension = len(grid)

    xgrid = ev(grid[0].lstrip())
    xpts = parseGrid(xgrid)
    ypts = np.array([0])
    if dimension == 2:
        ygrid = ev(grid[1].lstrip())
        ypts = parseGrid(ygrid)
    # build a sesame system
    system = Builder(xpts, ypts)
        
    # 2. set materials
    materials = settings['materials']
    for mat in materials:
        location = mat['location']
        # define a function that returns true/false dpending on location
        f = parseLocation(location, dimension)
        system.add_material(mat, f)
        # set the doping
        N_D = float(mat['N_D'])
        if N_D != 0:
            system.add_donor(N_D, f)
        N_A = float(mat['N_A'])
        if N_A != 0:
            system.add_acceptor(N_A, f)

    # 4. set the defects if present
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

        if isinstance(loc, float):
            system.add_defects(loc, N, se, sigma_h=sh, E=E, \
                                    transition=transition)
        elif len(loc) == 2:
            system.add_defects(loc, N, se, sigma_h=sh, E=E,\
                                    transition=transition)

    ##  ok i would go ahead and construct g for non-manual case here!
    lambda_power = np.array([])
    power = np.array([])
    lambda_alpha = np.array([])
    alpha = np.array([])
    if settings['use_manual_g'] is False:

        ########################################
        # illumination properties
        ########################################
        # use one sun power spectrum
        if settings['ill_onesun'] is True:
            lambda_power = onesundata[:,0]
            power = onesundata[:,1]* 1e-4  # converting to W/cm^2

        # read lambda and power
        if settings['ill_monochromatic'] is True:
            laserlambda = float(settings['ill_wavelength'])
            powertot = float(np.asarray(settings['ill_power']))
            # generate a distribution, Gaussian with fixed spread of 10 nm
            width = 100.
            lambda_power = np.linspace(280,4000,745)
            power = powertot/(2*np.pi*width**2)**.5 * np.exp(-(lambda_power-laserlambda)**2/(2*width**2))

        if not power.any():
            power = np.zeros(10)
            lambda_power = np.linspace(280,4000,745)

        ########################################
        # absorption properties
        ########################################
        if settings['abs_usefile'] is True:
            abs_file = settings['abs_file']
            lambda_alpha, alpha = parseAlphaFile(abs_file)
        if settings['abs_useralpha'] is True:
            alpha = np.asarray(settings['abs_alpha'])
            lambda_alpha = []
        if alpha.size == 0:
            alpha = np.zeros(745)
            lambda_alpha = np.linspace(280,4000,745)


    g = getgeneration(lambda_power, power, lambda_alpha, alpha, xpts)
    g = np.tile(g, system.ny)
    system.generation(g)
        

    
    return system

