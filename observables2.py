from numpy import exp
import numpy as np
from sesame.utils2 import get_dl


def get_n(sys, efn, v, sites):
    bl = 0
    return sys.Nc[sites] * exp(-bl+efn[sites]+v[sites])

def get_p(sys, efp, v, sites):
    bl = 0
    Eg = sys.Eg[sites]
    Nv = sys.Nv[sites]
    return Nv * exp(-Eg+bl+efp[sites]-v[sites])

def get_rr(sys, n, p, n1, p1, tau_e, tau_h, sites):
    ni = sys.ni[sites]
    r = (n*p - ni**2)/(tau_h * (n+n1) + tau_e*(p+p1))
    return r

def get_jn(sys, efn, v, sites, dl=None):
    # sites is a list of pairs of sites given in the folded representation
    bl = 0

    sp0, sp1 = [], []
    if dl is None:
        dl = []
    else:
        dl = dl
    for s in sites:
        sp0.append(s[0])
        sp1.append(s[1])
        if type(dl) == list:
            dl.append(get_dl(sys, s))
    dl = np.asarray(dl)

    vp0 = v[sp0]
    dv = vp0 - v[sp1]
    efnp0= efn[sp0]
    efnp1 = efn[sp1]

    Nc = sys.Nc[sp0]
    mu = sys.mu_e[sp0]

    dv = dv + (np.abs(dv) < 1e-5)*1e-5

    jn = exp(-bl) * (exp(efnp1) - exp(efnp0)) / dl * \
         dv / (-exp(-vp0)*(1 - exp(dv)))

    return jn * Nc * mu

def get_jp(sys, efp, v, sites, dl=None):
    bl = 0

    sp0, sp1 = [], []
    if dl is None:
        dl = []
    else:
        dl = dl
    for s in sites:
        sp0.append(s[0])
        sp1.append(s[1])
        if type(dl) == list:
            dl.append(get_dl(sys, s))
    dl = np.asarray(dl)

    vp0 = v[sp0]
    dv = vp0 - v[sp1]
    efpp0= efp[sp0]
    efpp1 = efp[sp1]

    Nv = sys.Nv[sp0]
    Eg = sys.Eg[sp0]
    mu = sys.mu_h[sp0]

    dv = dv + (np.abs(dv) < 1e-5)*1e-5

    jp = exp(-Eg + bl) * (exp(efpp1) - exp(efpp0)) / dl *\
         dv / (-exp(vp0)*(1 - exp(-dv)))

    return jp * Nv * mu

def get_jn_derivs(sys, efn, v, sites, dl=None):
    bl = 0

    sp0, sp1 = [], []
    if dl is None:
        dl = []
    else:
        dl = dl
    for s in sites:
        sp0.append(s[0])
        sp1.append(s[1])
        if type(dl) == list:
            dl.append(get_dl(sys, s))
    dl = np.asarray(dl)

    vp0 = v[sp0]
    vp1 = v[sp1]
    dv = vp0 - vp1
    efnp0= efn[sp0]
    efnp1 = efn[sp1]

    Nc = sys.Nc[sp0]
    mu = sys.mu_e[sp0]

    dv = dv + (np.abs(dv) < 1e-5)*1e-5
    defn_i = 1./dl * exp(-bl+efnp0) * (-dv)\
             / (-exp(-vp0)*(1 - exp(dv)))

    defn_ip1 = -1./dl * exp(-bl+efnp1) * (-dv)\
               / (-exp(-vp0) * (1 - exp(dv)))

    dv_i = -1./dl * exp(-bl) * (exp(efnp1) - exp(efnp0))\
           * exp(-vp0)*(1 + dv - exp(dv))\
           / (exp(-2*vp0) * (exp(dv)-1)**2)

    dv_ip1 = -1./dl * exp(-bl) * (exp(efnp1) - exp(efnp0))\
            * exp(-vp1) * (1 - dv - exp(-dv))\
            / (exp(-2*vp1) * (1-exp(-dv))**2)

    return mu*Nc*defn_i, mu*Nc*defn_ip1, mu*Nc*dv_i, mu*Nc*dv_ip1   

def get_jp_derivs(sys, efp, v, sites, dl=None):
    bl = 0

    sp0, sp1 = [], []
    if dl is None:
        dl = []
    else:
        dl = dl
    for s in sites:
        sp0.append(s[0])
        sp1.append(s[1])
        if type(dl) == list:
            dl.append(get_dl(sys, s))
    dl = np.asarray(dl)

    vp0 = v[sp0]
    vp1 = v[sp1]
    dv = vp0 - vp1
    efpp0= efp[sp0]
    efpp1 = efp[sp1]

    Nv = sys.Nv[sp0]
    Eg = sys.Eg[sp0]
    mu = sys.mu_h[sp0]

    dv = dv + (np.abs(dv) < 1e-5)*1e-5

    defp_i = 1/dl * exp(bl-Eg+efpp0) * (-dv)\
             / (-exp(vp0) * (1 - exp(-dv)))

    defp_ip1 = 1/dl * exp(bl-Eg+efpp1) * (-dv)\
               / (exp(vp0) * (1 - exp(-dv)))

    dv_i = -1/dl * exp(bl-Eg) * (exp(efpp1) - exp(efpp0))\
           * exp(vp0) * (1-dv - exp(-dv))\
           / (exp(2*vp0)*((exp(-dv)-1)**2))

    dv_ip1 = -1/dl * exp(bl-Eg) * (exp(efpp1) - exp(efpp0))\
            * exp(vp1) * (1+dv - exp(dv))\
            / (exp(2*vp1)*((1-exp(dv))**2))

    return mu*Nv*defp_i, mu*Nv*defp_ip1, mu*Nv*dv_i, mu*Nv*dv_ip1

def get_rr_derivs(sys, n, p, n1, p1, tau_e, tau_h, sites):
    ni = sys.ni[sites]

    defn = (n*p*(tau_h*(n+n1) + tau_e*(p+p1)) - (n*p-ni**2)*n*tau_h)\
         / (tau_h*(n+n1) + tau_e*(p+p1))**2
    defp = (n*p*(tau_h*(n+n1) + tau_e*(p+p1)) - (n*p-ni**2)*p*tau_e)\
         / (tau_h*(n+n1) + tau_e*(p+p1))**2
    dv = (n*p-ni**2) * (tau_h*p - tau_e*n) / (tau_h*(n+n1) + tau_e*(p+p1))**2

    return defn, defp, dv
