# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import sesame
import numpy as np
from numpy import exp
from PyQt5.QtCore import *
from PyQt5 import QtCore
import logging

logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

class SimulationWorker(QObject):

    def __init__(self, loop, system, solverSettings,\
                       generation, paramName, parent=None):
        super(SimulationWorker, self).__init__()

        self.parent = parent

        self.loop = loop
        self.system = system
        self.solverSettings = solverSettings
        self.generation = generation
        self.paramName = paramName

        self.logger = logging.getLogger(__name__)
 

    def __del__(self):
        self.wait()

    start = pyqtSignal(str)

    @pyqtSlot()
    def run(self):
        loop = self.loop
        system = self.system
        solverSettings = self.solverSettings
        generation = self.generation
        paramName = self.paramName

        # Solver settings
        loopValues, simName, fmt, BCs, contacts, Sc, tol, maxiter,\
                        useMumps, iterative = solverSettings

        # Add contacts surface recombination velocities
        system.contacts(*Sc)

        #===========================================================
        # Equilibrium potential
        #===========================================================
        self.logger.info("Solving for the equilibrium electrostatic potential")
        nx = system.nx
        # determine what the potential on the left and right might be
        if system.rho[0] < 0: # p-doped
            phi_left = -system.Eg[0] - np.log(abs(system.rho[0])/system.Nv[0])
        else: # n-doped
            phi_left = np.log(system.rho[0]/system.Nc[0])

        if system.rho[nx-1] < 0:
            phi_right = -system.Eg[nx-1] - np.log(abs(system.rho[nx-1])/system.Nv[nx-1])
            q = 1
        else:
            phi_right = np.log(system.rho[nx-1]/system.Nc[nx-1])
            q = -1

        # Make a linear guess and solve for the eqilibrium potential
        v = np.linspace(phi_left, phi_right, system.nx)
        if system.dimension == 2:
            v = np.tile(v, system.ny) # replicate the guess in the y-direction
        if system.dimension == 3:
            v = np.tile(v, system.ny*system.nz) # replicate the guess in the y and z-direction

        # Solver Poisson equation
        solution = {'v':v}
        solution = sesame.solve(system, solution, tol=tol, periodic_bcs=BCs,\
                                contacts_bcs=contacts, maxiter=maxiter,\
                                use_mumps=useMumps, iterative=iterative)

        if solution is not None:
            self.logger.info("Equilibrium electrostatic potential obtained")
        # Make a copy of the equilibrium potential
        veq = np.copy(solution['v'])
        # Initial arrays for the quasi-Fermi levels
        efn = np.zeros((system.nx*system.ny*system.nz,))
        efp = np.zeros((system.nx*system.ny*system.nz,))
        solution.update({'efn': efn, 'efp': efp})

        #===========================================================
        # Loop over voltages
        #===========================================================
        if loop == "voltage":
            self.logger.info("Voltage loop starting now")
            if generation != "":
                # create callable 
                if system.dimension == 1:
                    f = eval('lambda x:' + generation)
                elif system.dimension == 2:
                    f = eval('lambda x, y:' + generation)
                elif system.dimension == 3:
                    f = eval('lambda x, y, z:' + generation)
                # update generation rate of the system
                system.generation(f)

                # Loop at zero bias with increasing defect density of states
                self.logger.info("A generation rate is used. We are going to solve drift-diffusion-Poisson with increasing amplitudes of the generation rate to find a proper guess at zero-bias")
                system.g /= 1e10
                for a in range(10):
                    self.logger.info("Amplitude divided by {0}".format(1e10 / 10**a))
                    system.g *= 10**a
                    solution = sesame.solve(system, solution, equilibrium=veq, tol=tol,\
                                            periodic_bcs=BCs, maxiter=maxiter,\
                                            use_mumps=useMumps, iterative=iterative)
                system.g *= 10

            # Loop over voltages
            sesame.IVcurve(system, loopValues, solution, veq, simName, tol=tol,\
                           periodic_bcs=BCs, maxiter=maxiter, verbose=True,\
                           use_mumps=useMumps,\
                           iterative=iterative, fmt=fmt)
            if solution is not None:
                self.logger.info("********** Calculations completed **********")

        #===========================================================
        # Loop over generation rates
        #===========================================================
        if loop == 'generation':
            self.logger.info("Generation rate loop starting now")
            for idx, p in enumerate(loopValues):
                # give the named parameter its value
                exec(paramName + '=' + str(p))
                self.logger.info("Parameter value: {0} = {1}".format(paramName, p))
                # create callable 
                if system.dimension == 1:
                    f = eval('lambda x, {0}:'.format(paramName) + generation)
                elif system.dimension == 2:
                    f = eval('lambda x, y, {0}:'.format(paramName) + generation)
                elif system.dimension == 3:
                    f = eval('lambda x, y, z, {0}:'.format(paramName) + generation)
                # update generation rate of the system
                system.generation(f, args=(p,))

                system.g /= 1e10
                for a in range(11):
                    self.logger.info("Amplitude divided by {0}".format(1e10 / 10**a))
                    system.g *= 10**a
                    solution = sesame.solve(system, solution, equilibrium=veq, tol=tol,\
                                            periodic_bcs=BCs, maxiter=maxiter,\
                                            use_mumps=useMumps, iterative=iterative)
                if solution is not None:
                    name = fileName + "_{0}".format(idx)
                    if fmt == 'mat':
                        savemat(name, solution)
                    else:
                        np.savez(name, efn=solution['efn'], efp=solution['efp'],\
                                 v=solution['v'])
                else:
                    self.logger.info("The solver failed to converge for the parameter value"\
                          + " {0} (index {1}).".format(p, idx))
                    self.logger.info("Aborting now.")
                    exit(1)
            self.logger.info("********** Calculations completed **********")
