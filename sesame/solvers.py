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
        if use_mumps: 
            if mumps_available:
                spsolve = mumps.spsolve
            else:
                J = J.tocsr()
                warnings.warn('Could not import MUMPS. Default back to Scipy.'\
                              , UserWarning)
        else:
            spsolve = lg.spsolve
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

def get_rhs(x, sys, equilibrium, periodic_bcs, use_mumps):
    # Compute the right hand side of Ax=b
    if equilibrium:
        if periodic_bcs == False and sys.dimension != 1:
            rhs = importlib.import_module('.getFandJ_eq{0}_abrupt'\
                           .format(sys.dimension), 'sesame')
        else:
            rhs = importlib.import_module('.getFandJ_eq{0}'\
                           .format(sys.dimension), 'sesame')

        rhs, _ = rhs.getFandJ_eq(sys, x, use_mumps)
    else:
        if periodic_bcs == False and sys.dimension != 1:
            rhs = importlib.import_module('.getF{0}_abrupt'\
                           .format(sys.dimension), 'sesame')
        else:
            rhs = importlib.import_module('.getF{0}'\
                           .format(sys.dimension), 'sesame')

        rhs = rhs.getF(sys, x[2::3], x[0::3], x[1::3])

    return rhs


def get_jac(x, sys, equilibrium, periodic_bcs, use_mumps):
    # Compute the left hand side of Ax=b
    if equilibrium:
        if periodic_bcs == False and sys.dimension != 1:
            lhs = importlib.import_module('.getFandJ_eq{0}_abrupt'\
                           .format(sys.dimension), 'sesame')
        else:
            lhs = importlib.import_module('.getFandJ_eq{0}'\
                           .format(sys.dimension), 'sesame')

        _, lhs = lhs.getFandJ_eq(sys, x, use_mumps)
    else:
        if periodic_bcs == False and sys.dimension != 1:
            lhs = importlib.import_module('.jacobian{0}_abrupt'\
                           .format(sys.dimension), 'sesame')
        else:
            lhs = importlib.import_module('.jacobian{0}'\
                           .format(sys.dimension), 'sesame')

        lhs = lhs.getJ(sys, x[2::3], x[0::3], x[1::3], use_mumps)

    return lhs


def newton(sys, x, equilibrium, tol=1e-6, periodic_bcs=True,\
           maxiter=300, verbose=True, use_mumps=False,\
           iterative=False, inner_tol=1e-6, htp=1):
 

    htpy = np.linspace(1./htp, 1, htp)
    f0 = get_rhs(x, sys, equilibrium, periodic_bcs, use_mumps)

    for gdx, gamma in enumerate(htpy):
        if verbose:
            print("\nNewton loop {0}/{1}".format(gdx+1, htp))

        if gamma < 1:
            htol = 1
        else:
            htol = tol

        cc = 0
        converged = False

        while converged != True:
            cc = cc + 1
            # break if no solution found after maxiterations
            if cc > maxiter:
                print("Maximum number of iterations reached without solution: "\
                      + "no solution found!\n")
                break

            # solve linear system
            f = get_rhs(x, sys, equilibrium, periodic_bcs, use_mumps)
            f -= (1-gamma)*f0
            J = get_jac(x, sys, equilibrium, periodic_bcs, use_mumps)
            dx = sparse_solver(J, -f, use_mumps, iterative, inner_tol)
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

def solve(sys, guess, tol=1e-6, periodic_bcs=True, maxiter=300,\
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
    if 'efn' in guess.keys():
        x = np.zeros((3*sys.nx*sys.ny*sys.nz,), dtype=np.float64)
        x[0::3] = guess['efn']
        x[1::3] = guess['efp']
        x[2::3] = guess['v']

        x = newton(sys, x, False, tol=tol, periodic_bcs=periodic_bcs,\
                   maxiter=maxiter, verbose=verbose,\
                   use_mumps=use_mumps, iterative=iterative,\
                   inner_tol=inner_tol, htp=htp)
        if x is not None:
            x = {'efn': x[0::3], 'efp': x[1::3], 'v': x[2::3]}
    else:
        x = guess['v']

        x = newton(sys, x, True, tol=tol, periodic_bcs=periodic_bcs,\
                   maxiter=maxiter, verbose=verbose,\
                   use_mumps=use_mumps, iterative=iterative,\
                   inner_tol=inner_tol, htp=htp)
        if x is not None:
            x = {'v': x}

    return x


def IVcurve(sys, voltages, guess, file_name, tol=1e-6, periodic_bcs=True,\
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
