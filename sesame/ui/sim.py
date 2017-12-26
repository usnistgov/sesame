# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import numpy as np
from PyQt5.QtCore import *
from PyQt5 import QtCore
import logging

import sesame
from ..solvers import Solver

logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

class SimulationWorker(QObject):

    simuDone = pyqtSignal()
    newFile = pyqtSignal(str)

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
        self.abort = False
 
    @pyqtSlot()
    def abortSim(self):
        self.abort = True

    @pyqtSlot()
    def run(self):
        loop = self.loop
        system = self.system
        solverSettings = self.solverSettings
        generation = self.generation
        paramName = self.paramName

        # Solver settings
        loopValues, simName, fmt, BCs, contacts_bcs, contacts_WF, Sc, tol, maxiter,\
                        useMumps, iterative = solverSettings

        # Add contacts surface recombination velocities
        system.contacts(*Sc)

        # Create a Solver instance, I don't use the one already present
        solver = Solver()

        #===========================================================
        # Equilibrium potential
        #===========================================================
        nx = system.nx

        # Equilibrium guess
        guess = solver.make_guess(system, contacts_bcs=contacts_bcs,\
                                  contacts_WF=contacts_WF)
        # Solve Poisson equation
        solver.common_solver('Poisson', system, guess, tol, BCs, contacts_bcs,\
                              contacts_WF, maxiter, True, useMumps, iterative,\
                              1e-6, 1)

        if solver.equilibrium is not None:
            self.logger.info("Equilibrium electrostatic potential obtained")
            # Construct the solution dictionnary
            efn = np.zeros_like(solver.equilibrium)
            efp = np.zeros_like(solver.equilibrium)
            v = np.copy(solver.equilibrium)
            solution = {'efn': efn, 'efp': efp, 'v': v}
        else:
            self.logger.info("The solver failed to converge for the electrostatic potential")
            return

        if self.abort:
            self.simuDone.emit()
            return

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
                    system.g *= 10
                    solution = solver.common_solver('all', system, solution,\
                                    tol, BCs, contacts_bcs, contacts_WF, maxiter, True,\
                                    useMumps, iterative, 1e-6, 1)
                    if solution is None:
                        self.logger.info("The solver diverged. Aborting now.")
                        self.simuDone.emit()
                        return
                    if self.abort:
                        self.simuDone.emit()
                        return
            
            # Loop over voltages
            # sites of the right contact
            nx = system.nx
            s = [nx-1 + j*nx + k*nx*system.ny for k in range(system.nz)\
                                           for j in range(system.ny)]

            # sign of the voltage to apply
            if system.rho[nx-1] < 0:
                q = 1
            else:
                q = -1

            # Loop over the applied potentials made dimensionless
            Vapp = [i / system.scaling.energy for i in loopValues]
            for idx, vapp in enumerate(Vapp):
                logging.info("Applied voltage: {0} V".format(loopValues[idx]))

                # Apply the voltage on the right contact
                solution['v'][s] = solver.equilibrium[s] + q*vapp

                # Call the Drift Diffusion Poisson solver
                solution = solver.common_solver('all', system, solution,\
                                tol, BCs, contacts_bcs, contacts_WF, maxiter, True,\
                                useMumps, iterative, 1e-6, 1)
                if self.abort:
                    self.simuDone.emit()
                    return

                if solution is not None:
                    name = simName + "_{0}".format(idx)
                    # add some system settings to the saved results
                    solution.update({'x': system.xpts, 'y': system.ypts, \
                                     'z': system.zpts, 'affinity': system.bl,\
                                     'Eg': system.Eg, 'Nc': system.Nc,\
                                     'Nv': system.Nv,\
                                     'epsilon': system.epsilon})

                    if fmt == 'mat':
                        savemat(name, solution)
                    else:
                        np.savez_compressed(name, **solution)
                        # signal a new file has been created to the main thread
                        self.newFile.emit(name + '.npz')
                else:
                    logging.info("The solver failed to converge for the applied voltage"\
                          + " {0} V (index {1}).".format(loopValues[idx], idx))
                    self.logger.info("Aborting now.")
                    self.simuDone.emit()
                    return

            if solution is not None:
                self.logger.info("********** Calculations completed **********")

        #===========================================================
        # Loop over generation rates
        #===========================================================
        if loop == 'generation':
            self.logger.info("Generation rate loop starting now")
            for idx, p in enumerate(loopValues):
                # give the named parameter its value
                exec(paramName + '=' + str(p), globals())
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
                    system.g *= 10
                    solution = solver.common_solver('all', system, solution,\
                                    tol, BCs, contacts_bcs, contacts_WF, maxiter, True,\
                                    useMumps, iterative, 1e-6, 1)
                    if solution is None:
                        self.logger.info("The solver diverged. Aborting now.")
                        self.simuDone.emit()
                        return
                    if self.abort:
                        self.simuDone.emit()
                        return

                if solution is not None:
                    name = simName + "_{0}".format(idx)
                    # add some system settings to the saved results
                    solution.update({'x': system.xpts, 'y': system.ypts,\
                                     'z': system.zpts, 'affinity': system.bl,\
                                     'Eg': system.Eg, 'Nc': system.Nc,\
                                     'Nv': system.Nv, 'epsilon': system.epsilon})

                    if self.abort:
                        return
                    if fmt == 'mat':
                        savemat(name, solution)
                    else:
                        np.savez_compressed(name, **solution)
                        # signal a new file has been created to the main thread
                        self.newFile.emit(name + '.npz')
                else:
                    self.logger.info("The solver failed to converge for the parameter value"\
                          + " {0} (index {1}).".format(p, idx))
                    self.logger.info("Aborting now.")
                    self.simuDone.emit()
                    return
            if solution is not None:
                self.logger.info("********** Calculations completed **********")

        # tell main thread to quit this thread
        self.simuDone.emit()
