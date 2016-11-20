####################################
# Newton-Raphson algorithm
####################################
import sesame
import numpy as np
from mpi4py import MPI

from mumps import spsolve

def refine(dv):
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

def poisson_solver(sys, guess, tolerance, periodic_bcs=True, max_step=300, info=0):
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

    Returns
    -------
    v_final: numpy array of floats
        The final solution of the electrostatic potential in a one-dimensional
        array.
    """
    # import the module that create F and J
    if periodic_bcs == False and sys.dimension != 1:
        m = __import__('sesame.getFandJ_eq{0}_abrupt'.format(sys.dimension), globals(), locals(), ['getFandJ_eq'], 0)
    else:
        m = __import__('sesame.getFandJ_eq{0}'.format(sys.dimension), globals(), locals(), ['getFandJ_eq'], 0)

    getFandJ_eq = m.getFandJ_eq

    # first step of the Newton Raphson solver
    v = guess
    f, J = getFandJ_eq(sys, v)

    cc = 0
    clamp = 5.
    converged = False

    while converged != True:
        cc = cc + 1
        #-------- solve linear system ---------------------
        dx = spsolve(J, -f, MPI.COMM_WORLD)
        dx = dx.transpose()

        #--------- choose the new step -----------------
        error = max(np.abs(dx))

        if error < tolerance:
            converged = True
            v_final = v
            break 

        # use the usual clamping once a proper direction has been found
        elif error < 1e-3:
            # new correction and trial
            dv = dx / (1 + np.abs(dx/clamp))
            v = v + dv
            f, J = getFandJ_eq(sys, v)

            
        # Start slowly this refinement method found in a paper
        else:
            dv = refine(dx)
            v = v + dv
            f, J = getFandJ_eq(sys, v)

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



def ddp_solver(sys, guess, tolerance, periodic_bcs=True, max_step=300, info=0):
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

    Returns
    -------
    solution: dictionary
        Keys are 'efn', 'efp' and 'v'. The values contain one-dimensional numpy
        arrays of the solution.
    """

    # guess: initial guess passed to Newton Raphson algorithm
    # tolerance: max error accepted for delta u
    # max_step: maximum number of step allowed before declaring 'no solution
    # found'
    # info: integer, the program will print out the step number every 'info'
    # steps. If info is 0, no output is pronted out

    # import the module that create F and J
    if periodic_bcs == False and sys.dimension != 1:
        F = __import__('sesame.getF{0}_abrupt'.format(sys.dimension), globals(), locals(), ['getF'], 0)
        J = __import__('sesame.jacobian{0}_abrupt'.format(sys.dimension), globals(), locals(), ['getJ'], 0)

    else:
        F = __import__('sesame.getF{0}'.format(sys.dimension), globals(), locals(), ['getF'], 0)
        J = __import__('sesame.jacobian{0}'.format(sys.dimension), globals(), locals(), ['getJ'], 0)

    getF = F.getF
    getJ = J.getJ

    efn, efp, v = guess

    f = getF(sys, v, efn, efp)
    J = getJ(sys, v, efn, efp)
    solution = {'v': v, 'efn': efn, 'efp': efp}

    cc = 0
    clamp = 5.
    converged = False

    while converged != True:
        cc = cc + 1
        #-------- solve linear system ---------------------
        dx = spsolve(J, -f, MPI.COMM_WORLD)
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
        elif error < 1e-2:

            if error < 10: clamp = 10.
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

            f = getF(sys, v, efn, efp)
            J = getJ(sys, v, efn, efp)

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

            f = getF(sys, v, efn, efp)
            J = getJ(sys, v, efn, efp)

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
