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
from .. utils import save_sim
from scipy.io import savemat

logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

class SimulationWorker(QObject):

    simuDone = pyqtSignal()
    newFile = pyqtSignal(str)

    def __init__(self, loop, system, solverSettings,\
                       use_manual_g, generation, paramName, parent=None):
        super(SimulationWorker, self).__init__()

        self.parent = parent

        self.loop = loop
        self.system = system
        self.solverSettings = solverSettings
        self.generation = generation
        self.paramName = paramName
        self.use_manual_g = use_manual_g

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
        loopValues, simName, fmt, BCs, contacts_bcs, contacts_WF, Sc,\
        tol, maxiter, useMumps, iterative, ramp, iterPrec, htpy = solverSettings

        # Give contact type and add contacts surface recombination velocities
        left_type, right_type = contacts_bcs
        left_wf, right_wf = contacts_WF
        system.contact_type(left_type, right_type, left_wf, right_wf)
        system.contact_S(*Sc)

        # Create a Solver instance, I don't use the one already present
        solver = Solver(use_mumps=useMumps)

        #===========================================================
        # Equilibrium potential
        #===========================================================
        nx = system.nx

        # Equilibrium guess
        guess = solver.make_guess(system)
        # Solve Poisson equation
        solver.solve(system, 'Poisson', guess, tol, BCs, maxiter, True, htpy)

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
            self.logger.info("Nonequilibrium calculation starting now")

            if generation != "" and self.use_manual_g is True:
                # create callable 
                if system.dimension == 1:
                    f = eval('lambda x:' + generation)
                elif system.dimension == 2:
                    f = eval('lambda x, y:' + generation)
                # update generation rate of the system
                try:
                    system.generation(f)
                    has_generation = True
                except Exception:
                    msg = "**  The generation rate could not be interpreted  **"
                    self.logger.error(msg)
                    self.simuDone.emit()
                    return

            if self.use_manual_g is False and system.g.any():
                has_generation = True
            else:
                has_generation = False

            if has_generation is True:
                # Loop at zero bias with increasing defect density of states
                if ramp >  0:
                    self.logger.info("A generation rate is used with a non-zero ramp.")
                system.g /= 10**ramp
                for a in range(ramp):
                    self.logger.info("Amplitude divided by {0}"\
                                                .format(10**(ramp-a)))
                    solution = solver.solve(system, 'all', solution,\
                            tol, BCs, maxiter, True, htpy)
                    system.g *= 10 # the last one will be computed as part of
                                   # the voltage loop
                    if solution is None:
                        msg = "**  The calculations failed  **"
                        self.logger.error(msg)
                        self.simuDone.emit()
                        return
                    if self.abort:
                        self.simuDone.emit()
                        return
            
            # Loop over voltages
            # sites of the right contact
            nx = system.nx
            s = [nx-1 + j*nx  for j in range(system.ny)]

            # sign of the voltage to apply
            if system.rho[nx-1] < 0:
                q = 1
            else:
                q = -1

            # Loop over the applied potentials made dimensionless
            self.logger.info("Voltage loop starts now")
            Vapp = [i / system.scaling.energy for i in loopValues]
            for idx, vapp in enumerate(Vapp):
                logging.info("Applied voltage: {0} V".format(loopValues[idx]))

                # Apply the voltage on the right contact
                solution['v'][s] = solver.equilibrium[s] + q*vapp

                # Call the Drift Diffusion Poisson solver
                solution = solver.solve(system, 'all',solution,\
                                        tol, BCs, maxiter, True, htpy)
                if self.abort:
                    self.simuDone.emit()
                    return

                if solution is not None:
                    name = simName + "_{0}".format(idx)

                    if fmt == '.mat':
                        save_sim(system, solution, name, fmt='mat')
                    else:
                        filename = "%s.gzip" % name
                        save_sim(system, solution, filename)
                        # signal a new file has been created to the main thread
                        self.newFile.emit(name + '.gzip')
                else:
                    logging.info("The solver failed to converge for the applied voltage"\
                          + " {0} V (index {1}).".format(loopValues[idx], idx))
                    self.logger.info("Aborting now.")
                    self.simuDone.emit()
                    return

            if solution is not None:
                msg = "** Calculations completed successfully **"
                self.logger.info(msg)

        #===========================================================
        # Loop over generation rates
        #===========================================================
        if loop == 'generation':
            self.logger.info("Generation rate loop starting now")
            for idx, p in enumerate(loopValues):
                # give the named parameter its value
                self.logger.info("Parameter value: {0} = {1}".format(paramName, p))
                # create callable 
                if system.dimension == 1:
                    f = eval('lambda x, {0}:'.format(paramName) + generation)
                elif system.dimension == 2:
                    f = eval('lambda x, y, {0}:'.format(paramName) + generation)
                # update generation rate of the system
                try:
                    system.generation(f, args=(p,))
                except Exception:
                    msg = "**  The generation rate could not be interpreted  **"
                    self.logger.error(msg)
                    self.simuDone.emit()
                    return

                system.g /= 10**ramp
                for a in range(ramp+1):
                    self.logger.info("Amplitude divided by {0}"\
                                                .format(10**(ramp-a)))
                    solution = solver.solve(system, 'all', solution, tol, BCs, 
                                            maxiter, True, htpy)
                    system.g *= 10
                    if solution is None:
                        msg = "**  The calculations failed  **"
                        self.logger.error(msg)
                        self.simuDone.emit()
                        return
                    if self.abort:
                        self.simuDone.emit()
                        return
 
                if solution is not None:
                    name = simName + "_{0}".format(idx)
                    # add some system settings to the saved results
                    solution.update({'x': system.xpts, 'y': system.ypts,\
                                     'affinity': system.bl,\
                                     'Eg': system.Eg, 'Nc': system.Nc,\
                                     'Nv': system.Nv, 'epsilon': system.epsilon})

                    if self.abort:
                        return
                    if fmt == '.mat':
                        save_sim(system, solution, name, fmt='mat')
                    else:
                        filename = "%s.gzip" % name
                        save_sim(system, solution, filename)
                        # signal a new file has been created to the main thread
                        self.newFile.emit(name + '.gzip')
                else:
                    self.logger.info("The solver failed to converge for the parameter value"\
                          + " {0} (index {1}).".format(p, idx))
                    self.logger.info("Aborting now.")
                    self.simuDone.emit()
                    return
            if solution is not None:
                msg = "** Calculations completed successfully **"
                self.logger.info(msg)

        # tell main thread to quit this thread
        self.simuDone.emit()
