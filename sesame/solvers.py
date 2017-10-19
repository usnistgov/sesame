# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import numpy as np
import importlib
import warnings
from scipy.io import savemat

import scipy.sparse.linalg as lg
from scipy.sparse import spdiags

# check if MUMPS is available
mumps_available = False
try:
    from . import mumps
    mumps_available = True
except:
    pass


def damping(dx):
    # This damping procedure is inspired from Solid-State Electronics, vol. 19,
    # pp. 991-992 (1976).

    b = np.abs(dx) > 1
    dx[b] = np.log(1+np.abs(dx[b])*1.72)*np.sign(dx[b])


def sparse_solver(J, f, iterative, use_mumps, inner_tol):
    if not iterative:
        spsolve = lg.spsolve
        if use_mumps: 
            if mumps_available:
                spsolve = mumps.spsolve
            else:
                J = J.tocsr()
                warnings.warn('Could not import MUMPS. Default back to Scipy.'\
                              , UserWarning)
        dx = spsolve(J, f)
        return dx
    else:
        n = len(f)
        M = spdiags(1.0 / J.diagonal(), [0], n, n)
        dx, info = lg.lgmres(J, f, M=M, tol=inner_tol)
        if info == 0:
            return dx
        else:
            print("Iterative sparse solver failed with output info: ", info)
            exit(1)

def get_system(x, sys, equilibrium, periodic_bcs, use_mumps):
    # Compute the right hand side of J * x = f
    if equilibrium is None:
        if periodic_bcs == False and sys.dimension != 1:
            rhs = importlib.import_module('.getFandJ_eq{0}_abrupt'\
                           .format(sys.dimension), 'sesame')
        else:
            rhs = importlib.import_module('.getFandJ_eq{0}'\
                           .format(sys.dimension), 'sesame')

        f, J = rhs.getFandJ_eq(sys, x, use_mumps)
    else:
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
        J = lhs.getJ(sys, x[2::3], x[0::3], x[1::3], use_mumps)

    return f, J


def newton(sys, x, equilibrium=None, tol=1e-6, periodic_bcs=True,\
           maxiter=300, verbose=True, use_mumps=False,\
           iterative=False, inner_tol=1e-6, htp=1):

    htpy = np.linspace(1./htp, 1, htp)

    for gdx, gamma in enumerate(htpy):
        if verbose:
            print("\nNewton loop {0}/{1}".format(gdx+1, htp))

        if gamma < 1:
            htol = 1
        else:
            htol = tol

        cc = 0
        converged = False
        if gamma != 1:
            f0, _ = get_system(x, sys, equilibrium, periodic_bcs, use_mumps)

        while converged != True:
            cc = cc + 1
            # break if no solution found after maxiterations
            if cc > maxiter:
                print("Maximum number of iterations reached without solution: "\
                      + "no solution found!\n")
                break

            # solve linear system
            f, J = get_system(x, sys, equilibrium, periodic_bcs, use_mumps)
            if gamma != 1:
                f -= (1-gamma)*f0
            dx = sparse_solver(J, -f, iterative, use_mumps, inner_tol)
            dx.transpose()

            # compute error
            error = max(np.abs(dx))

            # damping and new value of x
            damping(dx)
            x += dx

            if error < htol:
                converged = True
                break 

            if np.isnan(error):
                print("The Newton solver diverged.")
                break

            # outputting status of solution procedure every so often
            if verbose:
                print('step {0}, error = {1}'.format(cc, error))
    if converged:
        return x
    else:
        return None

def solve(sys, guess, equilibrium=None, tol=1e-6, periodic_bcs=True, maxiter=300,\
          verbose=True, use_mumps=False, iterative=False, inner_tol=1e-6,\
          htp=1):
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
    # Solve for potential at equilibrium first no matter what
    if equilibrium is None:
        equilibrium = newton(sys, guess['v'], None, tol=tol, periodic_bcs=periodic_bcs,\
                      maxiter=maxiter, verbose=verbose,\
                      use_mumps=use_mumps, iterative=iterative,\
                      inner_tol=inner_tol, htp=htp)

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

        return x
    # If Efn is not provided, one only wants the equilibrium potential
    else:
        return {'v': equilibrium}


def IVcurve(sys, voltages, file_name, tol=1e-6, periodic_bcs=True,\
            maxiter=300, verbose=True, use_mumps=False,\
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
    nx = sys.nx

    # determine what the potential on the left and right might be
    if sys.rho[0] < 0: # p-doped
        phi_left = -sys.Eg[0] - np.log(abs(sys.rho[0])/sys.Nv[0])
    else: # n-doped
        phi_left = np.log(sys.rho[0]/sys.Nc[0])

    if sys.rho[nx-1] < 0:
        phi_right = -sys.Eg[nx-1] - np.log(abs(sys.rho[nx-1])/sys.Nv[nx-1])
        q = 1
    else:
        phi_right = np.log(sys.rho[nx-1]/sys.Nc[nx-1])
        q = -1

    # Make a linear guess and solve for the equilibrium potential
    v = np.linspace(phi_left, phi_right, sys.nx)
    if sys.dimension == 2:
        v = np.tile(v, sys.ny) # replicate the guess in the y-direction
    if sys.dimension == 3:
        v = np.tile(v, sys.ny*sys.nz) # replicate the guess in the y and z-direction

    if verbose:
        print("\nSolving for the equilibrium electrostatic potential...")
    phi_eq = newton(sys, v, tol=tol, periodic_bcs=periodic_bcs,\
                   maxiter=maxiter, verbose=verbose,\
                   use_mumps=use_mumps, iterative=iterative,\
                   inner_tol=inner_tol, htp=htp)
    if phi_eq is None:
        print("The solver failed to converge")
        print("Aborting now.")
        exit(1)   
    else:
        if verbose:
            print("\ndone")

    # create a dictionary 'result' with efn and efp
    efn = np.zeros((sys.nx*sys.ny*sys.nz,))
    efp = np.zeros((sys.nx*sys.ny*sys.nz,))
    result = {'v': np.copy(phi_eq), 'efn': efn, 'efp': efp}

    # sites of the right contact
    s = [nx-1 + j*nx + k*nx*sys.ny for k in range(sys.nz)\
                                   for j in range(sys.ny)]

    # Loop over the applied potentials made dimensionless
    Vapp = voltages / sys.scaling.energy
    for idx, vapp in enumerate(Vapp):

        if verbose:
            print("\napplied voltage: {0} V".format(voltages[idx]))

        # Apply the voltage on the right contact

        result['v'][s] = phi_eq[s] + q*vapp

        # Call the Drift Diffusion Poisson solver
        result = solve(sys, result, equilibrium=phi_eq, tol=tol, periodic_bcs=periodic_bcs,\
                       maxiter=maxiter, verbose=verbose,\
                       use_mumps=use_mumps, iterative=iterative,\
                       inner_tol=inner_tol, htp=htp)

        if result is not None:
            name = file_name + "_{0}".format(idx)
            if fmt == 'mat':
                savemat(name, result)
            else:
                np.savez(name, efn=result['efn'], efp=result['efp'],\
                         v=result['v'])
        else:
            print("The solver failed to converge for the applied voltage"\
                  + " {0} V (index {1}).".format(voltages[idx], idx))
            print("Aborting now.")
            exit(1)
