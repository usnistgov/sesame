# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import sys as osys
import numpy as np
import importlib
import warnings
from scipy.io import savemat

import scipy.sparse.linalg as lg
from scipy.sparse import spdiags
from scipy.sparse import coo_matrix, csr_matrix

import logging
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

# check if MUMPS is available
mumps_available = False
try:
    from . import mumps
    mumps_available = True
except:
    pass


class NewtonError(Exception):
    pass

class SparseSolverError(Exception):
    pass

class BCsError(Exception):
    def __init__(self, BCs):
        msg = "\n*********************************************" +\
              "\n*  Unknown contacts boundary conditions     *" +\
              "\n*********************************************"
        logging.error(msg)
        logging.error("Contacts boundary conditions: '{0}' is different from 'Dirichlet' or 'Neumann'.\n".format(BCs))

class SolverError(Exception):
    def __init__(self):
        msg = "\n*********************************************" +\
              "\n*       No solution could be found          *" +\
              "\n*********************************************"
        logging.error(msg)
        osys.exit(1)


def damping(dx):
    # This damping procedure is inspired from Solid-State Electronics, vol. 19,
    # pp. 991-992 (1976).

    b = np.abs(dx) > 1
    dx[b] = np.log(1+np.abs(dx[b])*1.72)*np.sign(dx[b])


def sparse_solver(J, f, iterative, use_mumps, inner_tol):
    if not iterative:
        spsolve = lg.spsolve
        if use_mumps and mumps_available: 
            spsolve = mumps.spsolve
        else:
            J = J.tocsr()
        dx = spsolve(J, f)
        return dx
    else:
        n = len(f)
        M = spdiags(1.0 / J.diagonal(), [0], n, n)
        dx, info = lg.lgmres(J, f, M=M, tol=inner_tol)
        if info == 0:
            return dx
        else:
            logging.info("Iterative sparse solver failed with output info: ".format(info))

def get_system(x, sys, equilibrium, periodic_bcs, contacts_bcs, use_mumps):
    # Compute the right hand side of J * x = f
    if equilibrium is None:
        size = sys.nx * sys.ny * sys.nz
        if sys.dimension != 1:
            rhs = importlib.import_module('.getFandJ_eq{0}'\
                           .format(sys.dimension), 'sesame')
            f, rows, columns, data = rhs.getFandJ_eq(sys, x, periodic_bcs, contacts_bcs)
        else:
            rhs = importlib.import_module('.getFandJ_eq1'\
                           .format(sys.dimension), 'sesame')
            f, rows, columns, data = rhs.getFandJ_eq(sys, x, contacts_bcs)

    else:
        size = 3 * sys.nx * sys.ny * sys.nz
        if periodic_bcs == False and sys.dimension != 1:
            rhs = importlib.import_module('.getF{0}_abrupt'\
                           .format(sys.dimension), 'sesame')
            lhs = importlib.import_module('.jacobian{0}_abrupt'\
                           .format(sys.dimension), 'sesame')
        else:
            rhs = importlib.import_module('.getF{0}'\
                           .format(sys.dimension), 'sesame')
            lhs = importlib.import_module('.jacobian{0}'\
                           .format(sys.dimension), 'sesame')

        f = rhs.getF(sys, x[2::3], x[0::3], x[1::3], equilibrium)
        rows, columns, data = lhs.getJ(sys, x[2::3], x[0::3], x[1::3])

    # form the Jacobian
    if use_mumps:
        J = coo_matrix((data, (rows, columns)), shape=(size, size), dtype=np.float64)
    else:
        J = csr_matrix((data, (rows, columns)), shape=(size, size), dtype=np.float64)

    return f, J


def newton(sys, x, equilibrium=None, tol=1e-6, periodic_bcs=True,\
           contacts_bcs='Dirichlet',
           maxiter=300, verbose=True, use_mumps=False,\
           iterative=False, inner_tol=1e-6, htp=1):

    htpy = np.linspace(1./htp, 1, htp)

    for gdx, gamma in enumerate(htpy):
        if verbose:
            logging.info("Newton loop {0}/{1}".format(gdx+1, htp))

        if gamma < 1:
            htol = 1
        else:
            htol = tol

        cc = 0
        converged = False
        if gamma != 1:
            f0, _ = get_system(x, sys, equilibrium, periodic_bcs, contacts_bcs, use_mumps)

        while converged != True:
            cc = cc + 1
            # break if no solution found after maxiterations
            if cc > maxiter:
                logging.error("Maximum number of iterations reached without solution: no solution found!")
                break

            # solve linear system
            f, J = get_system(x, sys, equilibrium, periodic_bcs,\
                              contacts_bcs, use_mumps)
            if gamma != 1:
                f -= (1-gamma)*f0

            try:
                dx = sparse_solver(J, -f, iterative, use_mumps, inner_tol)
                if dx is None:
                    raise SparseSolverError
                    break
                else:
                    dx.transpose()
                    # compute error
                    error = max(np.abs(dx))
                    if np.isnan(error) or error > 1e30:
                        raise NewtonError
                        break
                    if error < htol:
                        converged = True
                    else: 
                        # damping and new value of x
                        damping(dx)
                        x += dx
                        # print status of solution procedure every so often
                        if verbose:
                            logging.info('step {0}, error = {1}'.format(cc, error))
                            print('step {0}, error = {1}'.format(cc, error))
            except SparseSolverError:
                msg = "\n********************************************"+\
                      "\n*   The linear system could not be solved  *"+\
                      "\n********************************************"
                logging.error(msg)
                return None

            except NewtonError:
                msg = "\n********************************************"+\
                      "\n*  The Newton-Raphson algorithm diverged   *"+\
                      "\n********************************************"
                logging.error(msg)
                return None
                    
    if converged:
        return x
    else:
        return None



def solve(sys, guess, equilibrium=None, tol=1e-6, periodic_bcs=True,\
          contacts_bcs='Dirichlet', maxiter=300, verbose=True, use_mumps=False,\
          iterative=False, inner_tol=1e-6, htp=1):
    """
    Multi-purpose solver of Sesame.  If only the electrostatic potential is
    given as a guess, then the Poisson solver is used. If quasi-Fermi levels are
    passed, the Drift Diffusion Poisson solver is used.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    guess: dictionary of numpy arrays of floats
        Contains the one-dimensional arrays of the initial guesses for the
        electron quasi-Fermi level, the hole quasi-Fermi level and the
        electrostatic potential. Keys should be 'efn', 'efp' and 'v'.
    equilibrium: numpy array of floats
        Electrostatic potential of the system at thermal equilibrium. If not
        provided, the solver will solve for it before doing anything else.
    tol: float
        Accepted error made by the Newton-Raphson scheme.
    periodic_bcs: boolean
        Defines the choice of boundary conditions in the y-direction. True
        (False) corresponds to periodic (abrupt) boundary conditions.
    contacts_bcs: string
        Defines the choice of boundary conditions for the equilibrium electrostatic
        potential at the contact. 'Dirichlet' imposes the value of the potential
        given is the guess, 'Neumann' imposes a zero potential derivative.
    maxiter: integer
        Maximum number of steps taken by the Newton-Raphson scheme.
    verbose: boolean
        The solver returns the step number and the associated error at every
        step if set to True (default).
    use_mumps: boolean
        Defines if the MUMPS library should be used to solve for the Newton
        correction. Default is False.
    iterative: boolean
        Defines if an iterative method should be used to solve for the Newton
        correction instead of a direct method. Default is False.
    inner_tol: float
        Error of the inner iterative solver when used.
    htp: integer
        Number of homotopic Newton loops to perform.

    Returns
    -------

    solution: dictionary with  numpy arrays of floats or ``None``.
        Dictionary containing the one-dimensional arrays of the solution. The
        keys are the same as the ones for the guess. ``None`` is returned if no
        solution has been found.

    """
    # Solve for potential at equilibrium first if not provided
    if equilibrium is None:
        if not contacts_bcs in ['Dirichlet', 'Neumann']:
            raise BCsError(contacts_bcs)

        equilibrium = newton(sys, guess['v'], None, tol=tol,\
                              periodic_bcs=periodic_bcs,\
                              contacts_bcs=contacts_bcs,\
                              maxiter=maxiter, verbose=verbose,\
                              use_mumps=use_mumps, iterative=iterative,\
                              inner_tol=inner_tol, htp=htp)
        if equilibrium is None:
            raise SolverError

    # If Efn is provided, one wants a nonequilibrium solution 
    if 'efn' in guess.keys():
        x = np.zeros((3*sys.nx*sys.ny*sys.nz,), dtype=np.float64)
        x[0::3] = guess['efn']
        x[1::3] = guess['efp']
        x[2::3] = guess['v']

        x = newton(sys, x, equilibrium, tol=tol, periodic_bcs=periodic_bcs,\
                   maxiter=maxiter, verbose=verbose,\
                   use_mumps=use_mumps, iterative=iterative,\
                   inner_tol=inner_tol, htp=htp)
        if x is not None:
            x = {'efn': x[0::3], 'efp': x[1::3], 'v': x[2::3]}
        else:
            raise SolverError

    # If Efn is not provided, one only wants the equilibrium potential
    else:
        if equilibrium is not None:
            x = {'v': equilibrium}
    return x


def IVcurve(sys, voltages, guess, equilibrium, file_name, tol=1e-6,\
            periodic_bcs=True, maxiter=300, verbose=True, use_mumps=False,\
            iterative=False, inner_tol=1e-6, htp=1, fmt='npz'):
    """
    Solve the Drift Diffusion Poisson equations for the voltages provided. The
    results are stored in files with ``.npz`` format by default (See below for
    saving in Matlab format). Note that the
    potential is always applied on the right contact.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    voltages: array-like
        List of voltages for which the current should be computed.
    guess: dictionary of numpy arrays of floats
        Starting point of the solver. Keys of the dictionary must be 'efn',
        'efp', 'v' for the electron and quasi-Fermi levels, and the
        electrostatic potential respectively.
    equilibrium: numpy array of floats
        Electrostatic potential of the system at thermal equilibrium. If not
        provided, the solver will solve for it before doing anything else.
    file_name: string
        Name of the file to write the data to. The file name will be appended
        the index of the voltage list, e.g. ``file_name_0.npz``.
    tol: float
        Accepted error made by the Newton-Raphson scheme.
    periodic_bcs: boolean
        Defines the choice of boundary conditions in the y-direction. True
        (False) corresponds to periodic (abrupt) boundary conditions.
    maxiter: integer
        Maximum number of steps taken by the Newton-Raphson scheme.
    verbose: boolean
        The solver returns the step number and the associated error at every
        step, and this function prints the current applied voltage if set to True (default).
    use_mumps: boolean
        Defines if the MUMPS library should be used to solve for the Newton
        correction. Default is False.
    iterative: boolean
        Defines if an iterative method should be used to solve for the Newton
        correction instead of a direct method. Default is False.
    inner_tol: float
        Error of the inner iterative solver when used.
    htp: integer
        Number of homotopic Newton loops to perform.
    fmt: string
        Format string for the data files. Use ``mat`` to save the data in a
        Matlab format (version 5 and above).


    Notes
    -----
    The data files can be loaded and used as follows:

    >>> results = np.load('file.npz')
    >>> efn = results['efn']
    >>> efp = results['efp']
    >>> v = results['v']
    """
    # create a dictionary 'result' with efn and efp
    result = guess

    # sites of the right contact
    nx = sys.nx
    s = [nx-1 + j*nx + k*nx*sys.ny for k in range(sys.nz)\
                                   for j in range(sys.ny)]

    # sign of the voltage to apply
    if sys.rho[nx-1] < 0:
        q = 1
    else:
        q = -1

    # Loop over the applied potentials made dimensionless
    Vapp = voltages / sys.scaling.energy
    for idx, vapp in enumerate(Vapp):

        if verbose:
            logging.info("Applied voltage: {0} V".format(voltages[idx]))

        # Apply the voltage on the right contact
        result['v'][s] = equilibrium[s] + q*vapp

        # Call the Drift Diffusion Poisson solver
        result = solve(sys, result, equilibrium, tol=tol, periodic_bcs=periodic_bcs,\
                       maxiter=maxiter, verbose=verbose,\
                       use_mumps=use_mumps, iterative=iterative,\
                       inner_tol=inner_tol, htp=htp)

        if result is not None:
            name = file_name + "_{0}".format(idx)
            if fmt == 'mat':
                if (sys.ny > 1):
                    result.update(
                        {'x': sys.xpts, 'y': sys.ypts, 'chi': sys.bl, 'eg': sys.Eg, 'Nc': sys.Nc, 'Nv': sys.Nv})
                else:
                    result.update({'x': sys.xpts, 'chi': sys.bl, 'eg': sys.Eg, 'Nc': sys.Nc, 'Nv': sys.Nv})
                savemat(name, result)
            else:
                np.savez(name, efn=result['efn'], efp=result['efp'],\
                         v=result['v'])
        else:
            logging.error("The solver failed to converge for the applied voltage"\
                  + " {0} V (index {1}).".format(voltages[idx], idx))
            break
