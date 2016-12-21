import numpy as np
import importlib
import warnings

import scipy.sparse.linalg as lg
from scipy.sparse import spdiags

# check if MUMPS is available
mumps_available = False
try:
    from . import mumps
    mumps_available = True
except:
    pass


def refine(dx):
    # This damping procedure was taken from Solid-State Electronics, vol. 19,
    # pp. 991-992 (1976).

    a = np.abs(dx) < 1
    b = np.abs(dx) >= 3.7
    c = a == b # intersection of a and b

    dx[a] /= 2
    dx[b] = np.sign(dx[b]) * np.log(abs(dx[b]))/2
    dx[c] = np.sign(dx[c]) * abs(dx[c])**(0.2)/2

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

def poisson_solver(sys, guess, tol=1e-9, periodic_bcs=True, maxiter=300, 
                   eps=None, verbose=True, use_mumps=False, iterative=False):
    """
    Poisson solver of the system at thermal equilibrium.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    guess: numpy array of floats
        One-dimensional array of the guess for the electrostatic potential
        across the system.
    tol: float
        Accepted error made by the Newton-Raphson scheme.
    periodic_bcs: boolean
        Defines the choice of boundary conditions in the y-direction. True
        (False) corresponds to periodic (abrupt) boundary conditions.
    maxiter: integer
        Maximum number of steps taken by the Newton-Raphson scheme.
    eps: float
        Newton error above which a slow Newton convergence is chosen. The
        default is to use the fastest correction.
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
    v_final: numpy array of floats
        The final solution of the electrostatic potential in a one-dimensional
        array.
    """

    # import the module that create F and J
    if periodic_bcs == False and sys.dimension != 1:
        mod = importlib.import_module('.getFandJ_eq{0}_abrupt'.format(sys.dimension), 'sesame')
    else:
        mod = importlib.import_module('.getFandJ_eq{0}'.format(sys.dimension), 'sesame')


    # first step of the Newton Raphson solver
    v = guess

    cc = 0
    clamp = 5.
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
        dv = dv.transpose()

        # compute error
        error = max(np.abs(dv))

        if error < tol:
            converged = True
            v_final = v
            break 

        if eps is None or error < eps:
            # try to compute the next step with clamping method
            dv   = dv / (1 + np.abs(dv/clamp))
        elif eps is not None and error >= eps:
            # try a smaller refinement
            refine(dv)

        v = v + dv

        # outputing status of solution procedure every so often
        if verbose:
            print('step {0}, error = {1}'.format(cc, error))

    if converged:
        return v_final
    else:
        print("No solution found!\n")
        return None


def ddp_solver(sys, guess, tol=1e-9, periodic_bcs=True, maxiter=300,\
               eps=None, verbose=True, use_mumps=False, iterative=False):
    """
    Drift Diffusion Poisson solver of the system at out of equilibrium.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    guess: list [efn, efp, v] of numpy arrays of floats
        List of one-dimensional arrays of the initial guesses for the electron
        quasi-Fermi level (efn), the hole quasi-Fermi level (efp) and the
        electrostatic potential (v).
    tol: float
        Accepted error made by the Newton-Raphson scheme.
    periodic_bcs: boolean
        Defines the choice of boundary conditions in the y-direction. True
        (False) corresponds to periodic (abrupt) boundary conditions.
    maxiter: integer
        Maximum number of steps taken by the Newton-Raphson scheme.
    eps: float
        Newton error above which a slow Newton convergence is chosen. The
        default is to use the fastest correction.
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
    solution: dictionary or None
        Keys are 'efn', 'efp' and 'v'. The values contain one-dimensional numpy
        arrays of the solution. Returns ``None`` is failed to converge.
    """

    # import the module that create F and J
    if periodic_bcs == False and sys.dimension != 1:

        modF = importlib.import_module('.getF{0}_abrupt'.format(sys.dimension), 'sesame')
        modJ = importlib.import_module('.jacobian{0}_abrupt'.format(sys.dimension), 'sesame')
    else:
        modF = importlib.import_module('.getF{0}'.format(sys.dimension), 'sesame')
        modJ = importlib.import_module('.jacobian{0}'.format(sys.dimension), 'sesame')

    efn, efp, v = guess
    solution = {'v': v, 'efn': efn, 'efp': efp}

    cc = 0
    clamp = 5.
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
        dx = dx.transpose()

        # compute error
        error = max(np.abs(dx))

        if error < tol:
            converged = True
            solution['efn'] = efn
            solution['efp'] = efp
            solution['v']   = v
            break 

        if eps is None or error < eps:
            # try to compute the next step with clamping method
            defn = dx[0::3]
            defp = dx[1::3]
            dv   = dx[2::3]

            defn = dv + (defn - dv) / (1 + np.abs((defn-dv)/clamp))
            defp = dv + (defp - dv) / (1 + np.abs((defp-dv)/clamp))
            dv   = dv / (1 + np.abs(dv/clamp))
        elif eps is not None and error >= eps:
            # try a smaller refinement
            refine(dx)
            defn = dx[0::3]
            defp = dx[1::3]
            dv   = dx[2::3]

        # new values of efn, efp, v
        efn += defn
        efp += defp
        v += dv

        # outputing status of solution procedure every so often
        if verbose:
            print('step {0}, error = {1}'.format(cc, error))

    if converged:
        return solution
    else:
        print("No solution found!\n")
        return None
