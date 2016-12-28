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


def sparse_solver(J, f, iterative=False, use_mumps=False):
    if not iterative:
        spsolve = lg.spsolve
        if use_mumps: 
            if mumps_available:
                spsolve = mumps.spsolve
            else:
                J = J.tocsr()
                warnings.warn('Could not import MUMPS. Default back to Scipy.', UserWarning)
        dx = spsolve(J, f)
        return dx
    else:
        n = len(f)
        M = spdiags(1.0 / J.diagonal(), [0], n, n)
        tol = 1e-5
        dx, info = lg.lgmres(J, f, M=M, tol=tol)
        if info == 0:
            return dx
        else:
            print("Iterative sparse solver failed with output info: ", info)
            exit(1)

def poisson_solver(sys, guess, tol=1e-6, periodic_bcs=True, maxiter=300, 
                   verbose=True, use_mumps=False, iterative=False):
    """
    Poisson solver of the system at thermal equilibrium.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    guess: dictionary with one numpy array of floats
        Dictionary containing the one-dimensional array of the guess for the
        electrostatic potential across the system. The key should be ``'v'``.
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

    Returns
    -------

    solution: dictionary with one numpy array of floats or ``None``.
        Dictionary containing the one-dimensional array of the solution for the
        electrostatic potential across the system. The key is ``'v'``. ``None``
        is returned if no solution has been found.
    """

    # import the module that create F and J
    if periodic_bcs == False and sys.dimension != 1:
        mod = importlib.import_module('.getFandJ_eq{0}_abrupt'.format(sys.dimension), 'sesame')
    else:
        mod = importlib.import_module('.getFandJ_eq{0}'.format(sys.dimension), 'sesame')


    # first step of the Newton Raphson solver
    v = guess['v']

    cc = 0
    converged = False

    while converged != True:
        cc = cc + 1
        # break if no solution found after maxiterations
        if cc > maxiter:
            print('Poisson solver: too many iterations\n')
            break

        # solve linear system
        f, J = mod.getFandJ_eq(sys, v, use_mumps)
        dv = sparse_solver(J, -f, use_mumps=use_mumps, iterative=iterative)
        dv.transpose()

        # compute error
        error = max(np.abs(dv))

        if error < tol:
            converged = True
            solution = {'v': v}
            break 

        if error == np.nan:
            print("The Poisson solver diverged.")
            break

        # Newton correction damping
        damping(dv)
        v += dv

        # outputting status of solution procedure every so often
        if verbose:
            print('step {0}, error = {1}'.format(cc, error))

    if converged:
        return solution
    else:
        print("No solution found!\n")
        return None


def ddp_solver(sys, guess, tol=1e-6, periodic_bcs=True, maxiter=300,\
               verbose=True, use_mumps=False, iterative=False):
    """
    Drift Diffusion Poisson solver of the system out of equilibrium

    Parameters
    ----------
    sys: Builder
        The discretized system.
    guess: dictionary of numpy arrays of floats
        Contains the one-dimensional arrays of the initial guesses for the
        electron quasi-Fermi level, the hole quasi-Fermi level and the
        electrostatic potential. Keys should be 'efn', 'efp' and 'v'.
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

    Returns
    -------

    solution: dictionary with  numpy arrays of floats or ``None``.
        Dictionary containing the one-dimensional arrays of the solution. The
        keys are the same as the ones for the guess. ``None`` is returned if no
        solution has been found.
    """


    # import the module that create F and J
    if periodic_bcs == False and sys.dimension != 1:

        modF = importlib.import_module('.getF{0}_abrupt'.format(sys.dimension), 'sesame')
        modJ = importlib.import_module('.jacobian{0}_abrupt'.format(sys.dimension), 'sesame')
    else:
        modF = importlib.import_module('.getF{0}'.format(sys.dimension), 'sesame')
        modJ = importlib.import_module('.jacobian{0}'.format(sys.dimension), 'sesame')

    efn, efp, v = guess['efn'], guess['efp'], guess['v']

    cc = 0
    converged = False

    while converged != True:
        cc = cc + 1
        # break if no solution found after max iterations
        if cc > maxiter:
            print('Too many iterations\n')
            break

        # solve linear system
        f = modF.getF(sys, v, efn, efp)
        J = modJ.getJ(sys, v, efn, efp, use_mumps)
        dx = sparse_solver(J, -f, use_mumps=use_mumps, iterative=iterative)
        dx.transpose()

        # compute error
        error = max(np.abs(dx))

        if error < tol:
            converged = True
            solution = {'v': v, 'efn': efn, 'efp': efp}
            break 

        if error == np.nan:
            print("The drift diffusion Poisson solver diverged.")
            break

        # make sure that enormous corrections are damped
        damping(dx)
        efn += dx[0::3]
        efp += dx[1::3]
        v   += dx[2::3]

        # outputting status of solution procedure every so often
        if verbose:
            print('step {0}, error = {1}'.format(cc, error))
    
    if converged:
        return solution
    else:
        print("No solution found!\n")
        return None


def solve(sys, guess, tol=1e-6, periodic_bcs=True, maxiter=300,\
          verbose=True, use_mumps=False, iterative=False):
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

    Returns
    -------

    solution: dictionary with  numpy arrays of floats or ``None``.
        Dictionary containing the one-dimensional arrays of the solution. The
        keys are the same as the ones for the guess. ``None`` is returned if no
        solution has been found.

    """

    if 'efn' in guess.keys():
        solution = ddp_solver(sys, guess, tol=tol, periodic_bcs=periodic_bcs,\
                              maxiter=maxiter, verbose=verbose,\
                              use_mumps=use_mumps, iterative=iterative)
    else:
        solution = poisson_solver(sys, guess, tol=tol,\
                                  periodic_bcs=periodic_bcs,\
                                  maxiter=maxiter, verbose=verbose,\
                                  use_mumps=use_mumps, iterative=iterative)
    
    return solution


def IVcurve(sys, voltages, guess, file_name, tol=1e-6, periodic_bcs=True,\
            maxiter=300, verbose=True, use_mumps=False,\
            iterative=False, Matlab_format=False):
    """
    Solve the Drift Diffusion Poisson equations for the voltages provided. The
    results are stored in files with ``.npz`` format. Note that the potential
    is always applied on the right contact.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    voltages: array-like
        List of voltages for which the current should be computed.
    guess: dictionary of numpy arrays
        Starting point of the solver. Keys of the dictionary must be 'efn',
        'efp', 'v' for the electron and quasi-Fermi levels, and the
        electrostatic potential respectively.
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
    matlab_format: boolean
        Set the flag to true to save the data in a Matlab format (version 5 and
        above).


    Notes
    -----
    The data files can be loaded and used as follows:

    >>> results = np.load('file.npz')
    >>> efn = results['efn']
    >>> efp = results['efp']
    >>> v = results['v']
    """
    nx = sys.nx

    # determine what the potential on the left and right should be
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

    result = guess

    # sites of the right contact
    s = [nx-1 + j*nx + k*nx*sys.ny for k in range(sys.nz)\
                                   for j in range(sys.ny)]

    # Loop over the applied potentials made dimensionless
    Vapp = voltages / sys.scaling.energy
    for idx, vapp in enumerate(Vapp):

        if verbose:
            print("\napplied voltage: {0} V".format(voltages[idx]))

        # Apply the voltage on the right contact
        result['v'][s] = phi_right + q*vapp

        # Call the Drift Diffusion Poisson solver
        result = solve(sys, result, tol=tol, periodic_bcs=periodic_bcs,\
                       maxiter=maxiter, verbose=verbose,\
                       use_mumps=use_mumps, iterative=iterative)

        if result is not None:
            name = file_name + "_{0}".format(idx)
            if not Matlab_format:
                np.savez(name, efn=result['efn'], efp=result['efp'],\
                         v=result['v'])
            else:
                savemat(name, result)
        else:
            print("The ddp solver failed to converge for the applied voltage\
            {0} V (index {1})".format(voltages[idx], idx))
            print("I will abort now.")
            exit(1)
