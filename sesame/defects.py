# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.


import numpy as np
from scipy.integrate import quad
from scipy.constants import m_e, epsilon_0
from math import exp


def defectsF(sys, defects_list, n, p, rho, r=None):
    """
    These functions define the model for the charge at the defects.

    The functions we integrate are somewhat repetitive because I want to avoid
    making numerous Python function calls by quad.
    """


    for defect in defects_list:
        sites = defect.sites
        E = defect.energy
        a, b = max(defect.transition), min(defect.transition)

        # carrier densities
        _n = n[sites]
        _p = p[sites]
        ni2 = sys.ni[sites]**2
        _np = _n * _p

        # thermal velocity: arrays
        ### need to check unit type here!
        if sys.input_length=='m':
            ct = np.sqrt(epsilon_0/sys.scaling.density)/sys.scaling.mobility
        else:
            ct = 100*np.sqrt(epsilon_0*1e-2 / sys.scaling.density) / sys.scaling.mobility
        ve = ct * np.sqrt(3/(sys.mass_e[sites]*m_e)) 
        vh = ct * np.sqrt(3/(sys.mass_h[sites]*m_e))

        # capture cross setions: float
        se = defect.sigma_e
        sh = defect.sigma_h

        # density of states: float or function
        N = defect.dos
        dl = defect.perp_dl 

        if E is not None: # no integration, vectorize as much as possible
            _n1 = np.sqrt(sys.Nc[sites]*sys.Nv[sites]) * np.exp(-sys.Eg[sites]/2 + E)
            _p1 = np.sqrt(sys.Nc[sites]*sys.Nv[sites]) * np.exp(-sys.Eg[sites]/2 - E)

            # additional charge
            f = (se*ve*_n + sh*vh*_p1) / (se*ve*(_n+_n1) + sh*vh*(_p+_p1))
            rho[sites] += N / dl * (a + (b-a)*f)

            # additional recombination
            tau_e, tau_h = 1/(se*ve*N/dl), 1/(sh*vh*N/dl)
            if r is not None:
                r[sites] += (_np - ni2) / (tau_h*(_n+_n1) + tau_e*(_p+_p1))

        else: # integral to perform, quad requires single value function
            # additional recombination
            def _r(E, sdx, site):
                _n1 = np.sqrt(sys.Nc[site]*sys.Nv[site]) * np.exp(-sys.Eg[site]/2 + E)
                _p1 = np.sqrt(sys.Nc[site]*sys.Nv[site]) * np.exp(-sys.Eg[site]/2 - E)

                res = (_np[sdx] - ni2[sdx]) \
                      / ((_n[sdx]+_n1)/(sh*vh[sdx]) + (_p[sdx]+_p1)/(se*ve[sdx]))

                if not callable(N):
                    res *= N / dl[sdx]
                else: # if N is callable
                    res *= N(E) / dl[sdx]
                return res

            if r is not None:
                r[sites] += [quad(_r, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                  args=(sdx, s))[0] \
                             for sdx, s in enumerate(sites)]

            # additional charge
            def _rho(E, sdx, site):
                _n1 = np.sqrt(sys.Nc[site]*sys.Nv[site]) * np.exp(-sys.Eg[site]/2 + E)
                _p1 = np.sqrt(sys.Nc[site]*sys.Nv[site]) * np.exp(-sys.Eg[site]/2 - E)

                f = (se*ve[sdx]*_n[sdx] + sh*vh[sdx]*_p1) \
                  / (se*ve[sdx]*(_n[sdx]+_n1) + sh*vh[sdx]*(_p[sdx]+_p1))
                res = a + (b-a)*f

                if not callable(N):
                    res *= N / dl[sdx]
                else: # if N is callable
                    res *= N(E) / dl[sdx]
                return res

            rho[sites] += [quad(_rho, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                args=(sdx, s))[0] \
                           for sdx, s in enumerate(sites)]



            
def defectsJ(sys, defects_list, n, p, drho_dv, drho_defn=None, drho_defp=None,\
             dr_defn=None, dr_defp=None, dr_dv=None):


    for defect in defects_list:
        sites = defect.sites
        E = defect.energy
        a, b = max(defect.transition), min(defect.transition)

        # carrier densities
        _n = n[sites]
        _p = p[sites]
        ni2 = sys.ni[sites]**2
        _np = _n * _p

        # thermal velocity: arrays
        if sys.input_length=='m':
            ct = np.sqrt(epsilon_0/sys.scaling.density)/sys.scaling.mobility
        else:
            ct = 100*np.sqrt(epsilon_0*1e-2 / sys.scaling.density) / sys.scaling.mobility
        ve = ct * np.sqrt(3/(sys.mass_e[sites]*m_e)) 
        vh = ct * np.sqrt(3/(sys.mass_h[sites]*m_e))

        # capture cross setions: float
        se = defect.sigma_e
        sh = defect.sigma_h

        # density of states: float or function
        N = defect.dos
        dl = defect.perp_dl

        # multipurpose derivative of rho
        def drho(E, sdx, site, var):
            _n1 = np.sqrt(sys.Nc[site]*sys.Nv[site]) * np.exp(-sys.Eg[site]/2 + E)
            _p1 = np.sqrt(sys.Nc[site]*sys.Nv[site]) * np.exp(-sys.Eg[site]/2 - E)

            if var == 'v':
                res = (b-a) * ((se*ve[sdx])**2*_n[sdx]*_n1 +\
                                2*(sh*vh[sdx])*(se*ve[sdx])*_np[sdx]\
                                + (sh*vh[sdx])**2*_p[sdx]*_p1)

            if var == 'efn':
                res =  (b-a) * se*ve[sdx]*_n[sdx] * (se*ve[sdx]*_n1 + sh*vh[sdx]*_p[sdx])

            if var == 'efp':
                res = (b-a) * (se*ve[sdx]*_n[sdx] + sh*vh[sdx]*_p1) * sh*vh[sdx]*_p[sdx]

            res /= (se*ve[sdx]*(_n[sdx]+_n1)+sh*vh[sdx]*(_p[sdx]+_p1))**2

            if not callable(N):
                res *= N / dl[sdx]
            else: # if N is callable
                res *= N(E) / dl[sdx]
            return res

        # multipurpose derivative of recombination
        def dr(E, sdx, site, var):
            _n1 = np.sqrt(sys.Nc[site]*sys.Nv[site]) * np.exp(-sys.Eg[site]/2 + E)
            _p1 = np.sqrt(sys.Nc[site]*sys.Nv[site]) * np.exp(-sys.Eg[site]/2 - E)

            if var == 'efn':
                res = _np[sdx]*((_n[sdx]+_n1)/(sh*vh[sdx]) + (_p[sdx]+_p1)/(se*ve[sdx]))\
                       - (_np[sdx] - ni2[sdx])*_n[sdx]/(sh*vh[sdx])

            if var == 'efp':
                res = -_np[sdx]*((_n[sdx]+_n1)/(sh*vh[sdx]) + (_p[sdx]+_p1)/(se*ve[sdx]))\
                        - (_np[sdx] - ni2[sdx])*_p[sdx]/(se*ve[sdx])

            if var == 'v':
                res = (_np[sdx] - ni2[sdx]) * (_p[sdx]/(se*ve[sdx]) - _n[sdx]/(sh*vh[sdx]))

            res /= ((_n[sdx]+_n1)/(sh*vh[sdx]) + (_p[sdx]+_p1)/(se*ve[sdx]))**2

            if not callable(N):
                res *= N / dl[sdx]
            else: # if N is callable
                res *= N(E) / dl[sdx]
            return res

        # actual computation of things
        if E is not None: # no integration, vectorize as much as possible
            # additional charge
            _n1 = np.sqrt(sys.Nc[sites]*sys.Nv[sites]) * np.exp(-sys.Eg[sites]/2 + E)
            _p1 = np.sqrt(sys.Nc[sites]*sys.Nv[sites]) * np.exp(-sys.Eg[sites]/2 - E)

            d = (se*ve*(_n+_n1)+sh*vh*(_p+_p1))**2
            drho_dv[sites] += N/dl * (b-a) * ((se*ve)**2*_n*_n1 + 2*sh*vh*se*ve*_np\
                                           + (sh*vh)**2*_p*_p1) / d 

            if drho_defn is not None:
                drho_defn[sites] += N/dl * (b-a) * se*ve*_n * (se*ve*_n1 + sh*vh*_p) / d
                drho_defp[sites] += N/dl * (b-a) * (se*ve*_n + sh*vh*_p1) * sh*vh*_p / d
                
                tau_e, tau_h = 1/(se*ve*N/dl), 1/(sh*vh*N/dl)
                dr_defn[sites] += (_np*(tau_h*(_n+_n1) + tau_e*(_p+_p1)) - (_np-ni2)*_n*tau_h)\
                                  / (tau_h*(_n+_n1) + tau_e*(_p+_p1))**2
                dr_defp[sites] -= (_np*(tau_h*(_n+_n1) + tau_e*(_p+_p1)) - (_np-ni2)*_p*tau_e)\
                                  / (tau_h*(_n+_n1) + tau_e*(_p+_p1))**2
                dr_dv[sites]   += (_np-ni2) * (tau_e*_p - tau_h*_n)\
                                  / (tau_h*(_n+_n1) + tau_e*(_p+_p1))**2

        else: # integral to perform, quad requires single value function
            # always compute drho_dv
            drho_dv[sites] += [quad(drho, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                    args=(sdx, s, 'v'))[0] \
                               for sdx, s in enumerate(sites)]

            if drho_defn is not None:
                drho_defn[sites] += [quad(drho, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                          args=(sdx, s, 'efn'))[0] \
                                     for sdx, s in enumerate(sites)]

                drho_defp[sites] -= [quad(drho, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                          args=(sdx, s, 'efp'))[0] \
                                     for sdx, s in enumerate(sites)]

            if dr_defn is not None:
                dr_defn[sites] += [quad(dr, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                        args=(sdx, s, 'efn'))[0] \
                                   for sdx, s in enumerate(sites)]
                dr_defp[sites] -= [quad(dr, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                        args=(sdx, s, 'efp'))[0] \
                                   for sdx, s in enumerate(sites)]
                dr_dv[sites] += [quad(dr, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                      args=(sdx, s, 'v'))[0] \
                                 for sdx, s in enumerate(sites)]
