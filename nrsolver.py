####################################
# Newton-Raphson algorithm
####################################

import numpy as np
from scipy.sparse.linalg import spsolve

from sesame.getFandJ_eq import getFandJ_eq
from sesame.getFandJ import getFandJ

def solver(guess, tolerance, params, max_step=300, info=0):
    # guess: initial guess passed to Newton Raphson algorithm
    # tolerance: max error accepted for delta u
    # params: all physical parameters to pass to other functions
    # max_step: maximum number of step allowed before declaring 'no solution
    # found'
    # info: integer, the program will print out the step number every 'info'
    # steps. If info is 0, no output is pronted out

    if len(guess) == 1:
        thermal_eq = True
        v = guess[0]
        f, J = getFandJ_eq(v, params)
        solution = {'v': v}

    else:
        thermal_eq = False
        efn, efp, v = guess
        f, J = getFandJ(v, efn, efp, params)
        solution = {'v': v, 'efn': efn, 'efp': efp}

    cc = 0
    clamp = 5.
    converged = False
    while converged != True:
        cc = cc + 1
        new = spsolve(J, -f, use_umfpack=True)
        new = new.transpose()
        # getting the error of the guess
        error = max(np.abs(new))

        # if converged, then save data and break
        if error < tolerance:
            converged = True
            if thermal_eq:
                solution['v'] = v
            else:
                solution['efn'] = efn
                solution['efp'] = efp
                solution['v'] = v
            break 

        if thermal_eq:
            dv = new / (1 + np.abs(new/clamp))
            v += dv
            f, J = getFandJ_eq(v, params)

        else:
            # you can see how the variables are arranged: (efn, efp, v)
            defn = new[0::3]
            defp = new[1::3]
            dv = new[2::3]
            
            # it's necessary to "clamp" the size of the correction to improve
            # convergence
            defn = dv + (defn - dv) / (1 + np.abs((defn-dv)/clamp))
            defp = dv + (defp - dv) / (1 + np.abs((defp-dv)/clamp))
            dv = dv / (1 + np.abs(dv/clamp))

            # constructing the new Jacobian, and evaluation of f
            efn += defn
            efp += defp
            v += dv
            f, J = getFandJ(v, efn, efp, params)
        
        # outputing status of solution procedure every so often
        if info != 0 and np.mod(cc, info) == 0:
            print('step = {0}, error = {1}'.format(cc, error), "\n")

        # if no solution found after maxiterations, break
        if cc > max_step:
            print('too many iterations\n')
            break
    if converged:
        return solution
    else:
        print("No solution found!\n")
        return None
