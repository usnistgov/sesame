# These functions define the model for the charge a the defects.
#
# The functions we integrate are somewhat repetitive because I want to avoid
# making numerous Python function calls by quad

import numpy as np
from scipy.integrate import quad
from math import exp


def defectsF(sys, n, p, rho, r=None):

    for cdx, c in enumerate(sys.defects_types):
        sites = sys.extra_charge_sites[cdx]
        E = sys.defects_energies[cdx]
        a, b = max(c), min(c)

        # carrier densities
        _n = n[sites]
        _p = p[sites]
        ni2 = sys.ni[sites]**2
        _np = _n * _p

        # surface recombination velocities: float
        Se = sys.Seextra[cdx]
        Sh = sys.Shextra[cdx]

        # density of states: float or function
        N = sys.Nextra[cdx]

        if sys.dimension == 2:
            dl = sys.perp_dl[cdx]

        if E is not None: # no integration, vectorize as much as possible
            # additional charge
            _n1 = sys.Nc[sites] * np.exp(-sys.Eg[sites]/2 + E)
            _p1 = sys.Nv[sites] * np.exp(-sys.Eg[sites]/2 - E)

            if sys.dimension == 2:
                N = N / dl
                Se, Sh = Se / dl, Sh / dl

            rho[sites] += N * (a + (b-a)*(Se*_n + Sh*_p1)\
                             / (Se*(_n+_n1) + Sh*(_p+_p1)))

            # additional recombination
            if r is not None:
                r[sites] += (_np - ni2) / ((_n+_n1)/Sh + (_p+_p1)/Se)

        else: # integral to perform, quad requires single value function
            # additional recombination (density of states does not matter)
            def _r(E, sdx, site):
                _n1 = sys.Nc[site] * exp(-sys.Eg[site]/2 + E)
                _p1 = sys.Nv[site] * exp(-sys.Eg[site]/2 - E)

                res = (_np[sdx] - ni2[sdx]) \
                      / ((_n[sdx]+_n1) / Sh + (_p[sdx]+_p1) / Se)

                if sys.dimension == 2:
                    res /= dl[sdx]
                return res

            if r is not None:
                r[sites] += [quad(_r, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                  args=(sdx, s))[0] \
                             for sdx, s in enumerate(sites)]

            # additional charge
            def _rho(E, sdx, site):
                _n1 = sys.Nc[site] * exp(-sys.Eg[site]/2 + E)
                _p1 = sys.Nv[site] * exp(-sys.Eg[site]/2 - E)

                # dl scaling simplifies in occupancy
                f = (Se*_n[sdx] + Sh*_p1) / (Se*(_n[sdx]+_n1) + Sh*(_p[sdx]+_p1))
                res = a + (b-a)*f

                if not callable(N):
                    res *= N
                else: # if N is callable
                    res *= N(E)

                if sys.dimension == 2:
                    res /= dl[sdx]
                return res

            rho[sites] += [quad(_rho, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                args=(sdx, s))[0] \
                           for sdx, s in enumerate(sites)]



            
def defectsJ(sys, n, p, drho_dv, drho_defn=None, drho_defp=None,\
             dr_defn=None, dr_defp=None, dr_dv=None):

    for cdx, c in enumerate(sys.defects_types):
        sites = sys.extra_charge_sites[cdx]
        E = sys.defects_energies[cdx]
        a, b = max(c), min(c)

        # carrier densities
        _n = n[sites]
        _p = p[sites]
        ni2 = sys.ni[sites]**2
        _np = _n * _p

        # surface recombination velocities: float
        Se = sys.Seextra[cdx]
        Sh = sys.Shextra[cdx]

        # density of states: float or function
        N = sys.Nextra[cdx]
        n1 = sys.nextra[cdx]
        p1 = sys.pextra[cdx]

        if sys.dimension == 2:
            dl = sys.perp_dl[cdx]

        # multipurpose derivative of rho
        def drho(E, sdx, site, var):
            _n1 = sys.Nc[site] * exp(-sys.Eg[site]/2 + E)
            _p1 = sys.Nv[site] * exp(-sys.Eg[site]/2 - E)

            if var == 'v':
                res = (b-a) * (Se**2*_n[sdx]*_n1 + 2*Sh*Se*_np[sdx]\
                               + Sh**2*_p[sdx]*_p1)

            if var == 'efn':
                res =  (b-a) * Se*_n[sdx] * (Se*_n1 + Sh*_p[sdx])

            if var == 'efp':
                res = -(b-a) * (Se*_n[sdx] + Sh*_p1) * Sh*_p[sdx]

            res /= (Se*(_n[sdx]+_n1)+Sh*(_p[sdx]+_p1))**2

            if not callable(N):
                res *= N
            else: # if N is callable
                res *= N(E)

            if sys.dimension == 2:
                res /= dl[sdx]
            return res

        # multipurpose derivative of recombination
        def dr(E, sdx, site, var):
            _n1 = sys.Nc[site] * exp(-sys.Eg[site]/2 + E)
            _p1 = sys.Nv[site] * exp(-sys.Eg[site]/2 - E)

            if var == 'efn':
                res = _np[sdx]*((_n[sdx]+_n1)/Sh + (_p[sdx]+_p1)/Se)\
                       - (_np[sdx] - ni2[sdx])*_n[sdx]/Sh

            if var == 'efp':
                res = _np[sdx]*((_n[sdx]+_n1)/Sh + (_p[sdx]+_p1)/Se)\
                        - (_np[sdx] - ni2[sdx])*_p[sdx]/Se

            if var == 'v':
                res = (_np[sdx] - ni2[sdx]) * (_p[sdx]/Se - _n[sdx]/Sh)

            res /= ((_n[sdx]+_n1)/Sh + (_p[sdx]+_p1)/Se)**2

            if sys.dimension == 2:
                res /= dl[sdx]
            return res

        # actual computation of things
        if E is not None: # no integration, vectorize as much as possible
            # additional charge
            _n1 = sys.Nc[sites] * np.exp(-sys.Eg[sites]/2 + E)
            _p1 = sys.Nv[sites] * np.exp(-sys.Eg[sites]/2 - E)

            if sys.dimension == 2:
                N = N / dl
                Se, Sh = Se / dl, Sh / dl

            d = (Se*(_n+_n1)+Sh*(_p+_p1))**2
            drho_dv[sites] += N * (b-a) * (Se**2*_n*_n1 + 2*Sh*Se*_np\
                                           + Sh**2*_p*_p1) / d 

            if drho_defn is not None:
                drho_defn[sites] += N * (b-a) * Se*_n * (Se*_n1 + Sh*_p) / d
                drho_defp[sites] += -N * (b-a) * (Se*_n + Sh*_p1) * Sh*_p / d

                dr_defn[sites] += (_np*((_n+_n1)/Sh + (_p+_p1)/Se) - (_np-ni2)*_n/Sh)\
                                  / ((_n+_n1)/Sh + (_p+_p1)/Se)**2
                dr_defp[sites] += (_np*((_n+_n1)/Sh + (_p+_p1)/Se) - (_np-ni2)*_p/Se)\
                                  / ((_n+_n1)/Sh + (_p+_p1)/Se)**2
                dr_dv[sites]   += (_np-ni2) * (_p/Se - _n/Sh)\
                                  / ((_n+_n1)/Sh + (_p+_p1)/Se)**2

        else: # integral to perform, quad requires single value function
            # always compute drho_dv
            drho_dv[sites] += [quad(drho, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                    args=(sdx, s, 'v'))[0] \
                               for sdx, s in enumerate(sites)]

            if drho_defn is not None:
                drho_defn[sites] += [quad(drho, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                          args=(sdx, s, 'efn'))[0] \
                                     for sdx, s in enumerate(sites)]

                drho_defp[sites] += [quad(drho, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                          args=(sdx, s, 'efp'))[0] \
                                     for sdx, s in enumerate(sites)]

            if dr_defn is not None:
                dr_defn[sites] += [quad(dr, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                        args=(sdx, s, 'efn'))[0] \
                                   for sdx, s in enumerate(sites)]
                dr_defp[sites] += [quad(dr, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                        args=(sdx, s, 'efp'))[0] \
                                   for sdx, s in enumerate(sites)]
                dr_dv[sites] += [quad(dr, -sys.Eg[s]/2., sys.Eg[s]/2.,\
                                      args=(sdx, s, 'v'))[0] \
                                 for sdx, s in enumerate(sites)]
