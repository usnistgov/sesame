# These functions define the model for the charge a the defects.
from .observables import get_rr, get_rr_derivs

def defectsF(sys, n, p, rho, r=None):

    for cdx, c in enumerate(sys.defects_types):
        sites = sys.extra_charge_sites[cdx]

        nextra = sys.nextra[cdx]
        pextra = sys.pextra[cdx]
        _n = n[sites]
        _p = p[sites]

        Se = sys.Seextra[cdx]
        Sh = sys.Shextra[cdx]
        f = (Se*_n + Sh*pextra) / (Se*(_n+nextra) + Sh*(_p+pextra))

        # extra recombination
        if r is not None:
            r[sites] += get_rr(sys, _n, _p, nextra, pextra, 1/Se, 1/Sh, sites) 

        # extra charge
        if c == 'donor':
            rho[sites] += sys.Nextra[cdx] * (1 - f)
        if c == 'acceptor':
            rho[sites] -= sys.Nextra[cdx] * f
        if c == 'u-center':
            rho[sites] += sys.Nextra[cdx] * (1./2. - f)


            
def defectsJ(sys, n, p, drho_dv, drho_defn=None, drho_defp=None,\
             dr_defn=None, dr_defp=None, dr_dv=None):

    for cdx, c in enumerate(sys.defects_types):
        sites = sys.extra_charge_sites[cdx]

        nextra = sys.nextra[cdx]
        pextra = sys.pextra[cdx]
        _n = n[sites]
        _p = p[sites]

        Se = sys.Seextra[cdx]
        Sh = sys.Shextra[cdx]
        f = (Se*_n + Sh*pextra) / (Se*(_n+nextra) + Sh*(_p+pextra))

        d = (Se*(_n+nextra)+Sh*(_p+pextra))**2
        drho_dv[sites] += - sys.Nextra[cdx] *\
            (Se**2*_n*nextra + 2*Sh*Se*_p*_n + Sh**2*_p*pextra) / d
        if drho_defn is not None:
            drho_defn[sites] += - sys.Nextra[cdx] *\
                Se*_n * (Se*nextra + Sh*_p) / d
            drho_defp[sites] += sys.Nextra[cdx] *\
                (Se*_n + Sh*pextra) * Sh*_p / d

        if dr_defn is not None:
            defn, defp, dv =  get_rr_derivs(sys, _n, _p, nextra, pextra,\
                                            1/Se, 1/Sh, sites)
            dr_defn[sites] += defn
            dr_defp[sites] += defp
            dr_dv[sites]   += dv

