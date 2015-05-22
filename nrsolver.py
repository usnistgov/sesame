####################################
# Newton-Raphson algorithm
####################################

import numpy as np

from sesame.getFandJ_eq import getFandJ_eq
from sesame.getF import getF
from sesame.jacobian import getJ

# from scipy.sparse.linalg import spsolve
from mumps import spsolve
import mumps

def refine(dv):
    for sdx, s in enumerate(dv):
        if abs(s) < 1:
            dv[sdx] /= 2
        if 1 < abs(s) < 3.7:
            dv[sdx] = np.sign(s) * abs(s)**(0.2)/2
        elif abs(s) >= 3.7:
            dv[sdx] = np.sign(s) * np.log(abs(s))/2
    return dv

def solver(guess, tolerance, comm, params, max_step=300, info=0):
    # guess: initial guess passed to Newton Raphson algorithm
    # tolerance: max error accepted for delta u
    # comm: global communicator for parallel sparse solver
    # params: all physical parameters to pass to other functions
    # max_step: maximum number of step allowed before declaring 'no solution
    # found'
    # info: integer, the program will print out the step number every 'info'
    # steps. If info is 0, no output is pronted out

    rank = comm.Get_rank()

    if len(guess) == 1:
        thermal_eq = True
        v = guess[0]
        f, J = getFandJ_eq(v, params)
        solution = {'v': v}

    else:
        thermal_eq = False
        efn, efp, v = guess
        efno, efpo, vo = guess
        f = getF(v, efn, efp, params)
        J = getJ(v, efn, efp, params)
        solution = {'v': v, 'efn': efn, 'efp': efp}

    cc = 0
    clamp = 5.
    converged = False

    while converged != True:
        cc = cc + 1
        #-------- solve linear system ---------------------
        dx = spsolve(J, -f)

        # ctx = mumps.DMumpsContext(sym=0, par=1, comm=comm)
        # if ctx.myid == 0:
        #     ctx.set_centralized_sparse(J.tocoo())
        #     x = (-f).copy()
        #     ctx.set_rhs(x)
        #
        # # Silence most messages
        # ctx.set_silent()
        #
        # ctx.set_icntl(7, 3)
        #
        # # Analysis + Factorization + Solve
        # ctx.run(job=6)
        # ctx.destroy()
        #
        # if rank == 0:
        #     dx = x
        # else:
        #     dx = None
        # dx = comm.bcast(dx, root=0)

        dx = dx.transpose()

        #--------- choose the new step -----------------
        error = max(np.abs(dx))

        if error < tolerance:
            converged = True
            if thermal_eq:
                solution['v'] = v
            else:
                solution['efn'] = efn
                solution['efp'] = efp
                solution['v'] = v
            break 

        # use the usual clamping once a proper direction has been found
        elif error < 1e-3:
            if thermal_eq:
                # new correction and trial
                dv = dx / (1 + np.abs(dx/clamp))
                v = v + dv
                f, J = getFandJ_eq(v, params)
            else:
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

                f = getF(v, efn, efp, params)
                J = getJ(v, efn, efp, params)

        # Start slowly this refinement method found in a paper
        else:
            if thermal_eq:
                dv = refine(dx)
                v = v + dv
                f, J = getFandJ_eq(v, params)
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

                f = getF(v, efn, efp, params)
                J = getJ(v, efn, efp, params)


        # outputing status of solution procedure every so often
        if info != 0 and np.mod(cc, info) == 0 and rank == 0:
            print('step = {0}, error = {1}'.format(cc, error), "\n")

        # if no solution found after maxiterations, break
        if cc > max_step and rank == 0:
            print('too many iterations\n')
            break

    if converged:
        return solution
    else:
        print("No solution found!\n")
        return None
