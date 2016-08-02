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

    from sesame.observables import get_jn, get_jp, get_n, get_p, get_rr
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    def integrator(sys, v, efn, efp, sites_i, sites_ip1, dl):
        # return the current in the x-direction, summed along the y-axis
        jn = get_jn(sys, efn, v, sites_i, sites_ip1, dl)
        jp = get_jp(sys, efp, v, sites_i, sites_ip1, dl)
        return jn + jp

    while converged != True:
        cc = cc + 1
        #-------- solve linear system ---------------------
        dx = spsolve(J, -f, MPI.COMM_WORLD)
        dx = dx.transpose()

        #--------- choose the new step -----------------
        error = max(np.abs(dx))
        # J = integrator(sys, v, efn, efp, 10, 11, sys.dx[10])
        # sites = [i for i in range(sys.nx)]
        # p = get_p(sys, efp, v, sites)
        # n = get_n(sys, efp, v, sites)
        # r = get_rr(sys, n, p, sys.n1[0], sys.p1[0], sys.tau_e[0], sys.tau_h[0], sites)
        # sp = spline(sys.xpts, r)
        # x = sys.xpts
        # gtot = spline(x, sys.g).integral(x[0], x[-1])
        # print(J/gtot+sp.integral(0, sys.xpts[-1])/gtot)

       
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
