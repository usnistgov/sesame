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



def refine( dv):
    # This damping procedure was taken from Solid-State Electronics, vol. 19,
    # pp. 991-992 (1976).
    for sdx, s in enumerate(dv):
        if abs(s) < 1:
            dv[sdx] /= 2
        if 1 < abs(s) < 3.7:
            dv[sdx] = np.sign(s) * abs(s)**(0.2)/2
        elif abs(s) >= 3.7:
            dv[sdx] = np.sign(s) * np.log(abs(s))/2
    return dv

def sparse_solver(J, f, iterative=False, use_mumps=False):
    if not iterative:
        spsolve = lg.spsolve
        if use_mumps: 
            if mumps_available:
                spsolve = mumps.spsolve
            else:
                J = J.tocsc()
                warnings.warn('Could not import MUMPS. Default back to Scipy.', UserWarning)
        dx = spsolve(J, f)
        return dx
    else:
        n = len(f)
        # Better than Scipy incomplete LU but not better than 1/diag(J)
        # import pyamg
        # ml = pyamg.smoothed_aggregation_solver(J)
        # M = ml.aspreconditioner()
        M = spdiags(1.0 / J.diagonal(), [0], n, n)

        dx, info = lg.lgmres(J, f, M=M)

        if info == 0:
            return dx
        else:
            print("Iterative sparse solver failed with output info: ", info)
            exit(1)

def poisson_solver(sys, guess, tolerance=1e-9, periodic_bcs=True, max_step=300, 
                   info=1, use_mumps=False, iterative=False):
    """
    Poisson solver of the system at thermal equilibrium.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    guess: numpy array of floats
        One-dimensional array of the guess for the electrostatic potential
        across the system.
    tolerance: float
        Accepted error made by the Newton-Raphson scheme.
    periodic_bcs: boolean
        Defines the choice of boundary conditions in the y-direction. True
        (False) corresponds to periodic (abrupt) boundary conditions.
    max_step: integer
        Maximum number of steps taken by the Newton-Raphson scheme.
    info: integer
        The solver returns the step number and the associated error every info
        steps.
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
    f, J = mod.getFandJ_eq(sys, v, use_mumps)

    cc = 0
    clamp = 5.
    converged = False

    while converged != True:
        cc = cc + 1
        #-------- solve linear system ---------------------
        dx = sparse_solver(J, -f, use_mumps=use_mumps, iterative=iterative)
        dx = dx.transpose()

        #--------- choose the new step -----------------
        error = max(np.abs(dx))

        if error < tolerance:
            converged = True
            v_final = v
            break 

        # use the usual clamping once a proper direction has been found
        elif error < 1:
            # new correction and trial
            dv = dx / (1 + np.abs(dx/clamp))
            v = v + dv
            f, J = mod.getFandJ_eq(sys, v, use_mumps)
            
        # Start slowly this refinement method found in a paper
        else:
            dv = refine(dx)
            v = v + dv
            f, J = mod.getFandJ_eq(sys, v, use_mumps)

        # outputing status of solution procedure every so often
        if info != 0 and np.mod(cc, info) == 0:
            print('step = {0}, error = {1}'.format(cc, error))

        # if no solution found after maxiterations, break
        if cc > max_step:
            print('Poisson solver: too many iterations\n')
            break

    if converged:
        return v_final
    else:
        print("No solution found!\n")
        return None


def ddp_solver(sys, guess, tolerance=1e-9, periodic_bcs=True, max_step=300,\
               info=1, use_mumps=False, iterative=False):
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
    tolerance: float
        Accepted error made by the Newton-Raphson scheme.
    periodic_bcs: boolean
        Defines the choice of boundary conditions in the y-direction. True
        (False) corresponds to periodic (abrupt) boundary conditions.
    max_step: integer
        Maximum number of steps taken by the Newton-Raphson scheme.
    info: integer
        The solver returns the step number and the associated error every info
        steps.
    with_mumps: Boolean
        Flag to decide whether to use MUMPS library or not. Default is False.
    use_mumps: boolean
        Defines if the MUMPS library should be used to solve for the Newton
        correction. Default is False.
    iterative: boolean
        Defines if an iterative method should be used to solve for the Newton
        correction instead of a direct method. Default is False.

    Returns
    -------
    solution: dictionary
        Keys are 'efn', 'efp' and 'v'. The values contain one-dimensional numpy
        arrays of the solution.
    """

    # import the module that create F and J
    if periodic_bcs == False and sys.dimension != 1:

        modF = importlib.import_module('.getF{0}_abrupt'.format(sys.dimension), 'sesame')
        modJ = importlib.import_module('.jacobian{0}_abrupt'.format(sys.dimension), 'sesame')
    else:
        modF = importlib.import_module('.getF{0}'.format(sys.dimension), 'sesame')
        modJ = importlib.import_module('.jacobian{0}'.format(sys.dimension), 'sesame')

    efn, efp, v = guess

    f = modF.getF(sys, v, efn, efp)
    J = modJ.getJ(sys, v, efn, efp, use_mumps)
    solution = {'v': v, 'efn': efn, 'efp': efp}

    cc = 0
    clamp = 5.
    converged = False

    while converged != True:
        cc = cc + 1
        #-------- solve linear system ---------------------
        dx = sparse_solver(J, -f, use_mumps=use_mumps, iterative=iterative)
        dx = dx.transpose()

        #--------- choose the new step -----------------
        error = max(np.abs(dx))
       
        if error < tolerance:
            converged = True
            solution['efn'] = efn
            solution['efp'] = efp
            solution['v'] = v
            break 

        # use the usual clamping once a proper direction has been found
        elif error < 1:
            # you can see how the variables are arranged: (efn, efp, v)
            defn = dx[0::3]
            defp = dx[1::3]
            dv = dx[2::3]

            defn = dv + (defn - dv) / (1 + np.abs((defn-dv)/clamp))
            defp = dv + (defp - dv) / (1 + np.abs((defp-dv)/clamp))
            dv = dv / (1 + np.abs(dv/clamp))

            efn = efn + defn
            efp = efp + defp
            v = v + dv

            f = modF.getF(sys, v, efn, efp)
            J = modJ.getJ(sys, v, efn, efp, use_mumps)

        # Start slowly with this refinement method found in a paper
        else:
            # you can see how the variables are arranged: (efn, efp, v)
            defn = dx[0::3]
            defp = dx[1::3]
            dv = dx[2::3]

            defn = refine(defn)
            defp = refine(defp)
            dv = refine(dv)

            efn = efn + defn
            efp = efp + defp
            v = v + dv

            f = modF.getF(sys, v, efn, efp)
            J = modJ.getJ(sys, v, efn, efp, use_mumps)

        # outputing status of solution procedure every so often
        if info != 0 and np.mod(cc, info) == 0:
            print('step = {0}, error = {1}'.format(cc, error))

        # if no solution found after maxiterations, break
        if cc >= max_step:
            print('too many iterations\n')
            break

    if converged:
        return solution
    else:
        print("No solution found!\n")
        return None
