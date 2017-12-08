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
            phi_left = -system.Eg[0] - np.log(abs(system.rho[0])/system.Nv[0]) - system.bl[0]
        else: # n-doped
            phi_left = np.log(system.rho[0]/system.Nc[0]) - system.bl[0]

        if system.rho[nx-1] < 0:
            phi_right = -system.Eg[nx-1] - np.log(abs(system.rho[nx-1])/system.Nv[nx-1]) - system.bl[nx-1]
            q = 1
        else:
            phi_right = np.log(system.rho[nx-1]/system.Nc[nx-1]) - system.bl[nx-1]
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
        else:
            self.logger.info("The solver failed to converge for the electrostatic potential")
            return

        if self.abort:
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
                    solution = sesame.solve(system, solution, equilibrium=veq, tol=tol,\
                                            periodic_bcs=BCs, maxiter=maxiter,\
                                            use_mumps=useMumps, iterative=iterative)
                    if solution is None:
                        self.logger.info("The solver diverged. Aborting now.")
                        return
                    if self.abort:
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
                solution['v'][s] = veq[s] + q*vapp

                # Call the Drift Diffusion Poisson solver
                solution = sesame.solve(system, solution, equilibrium=veq, tol=tol,\
                                            periodic_bcs=BCs, maxiter=maxiter,\
                                            use_mumps=useMumps, iterative=iterative)

                if self.abort:
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
                    self.newFile.emit(name)
                else:
                    logging.info("The solver failed to converge for the applied voltage"\
                          + " {0} V (index {1}).".format(loopValues[idx], idx))
                    break

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
                    system.g *= 10
                    solution = sesame.solve(system, solution, equilibrium=veq, tol=tol,\
                                            periodic_bcs=BCs, maxiter=maxiter,\
                                            use_mumps=useMumps, iterative=iterative)
                    if solution is None:
                        self.logger.info("The solver diverged. Aborting now.")
                        break
                    if self.abort:
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
                    self.newFile.emit(name)
                else:
                    self.logger.info("The solver failed to converge for the parameter value"\
                          + " {0} (index {1}).".format(p, idx))
                    self.logger.info("Aborting now.")
                    break
            self.logger.info("********** Calculations completed **********")

        # tell main thread to quit this thread
        self.simuDone.emit()
