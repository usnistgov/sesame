from numpy import exp
import numpy as np


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

def get_jn(sys, efn, v, sites_i, sites_ip1, dl):
    # sites is a list of pairs of sites given in the folded representation
    bl = 0

    vp0 = v[sites_i]
    dv = vp0 - v[sites_ip1]
    efnp0= efn[sites_i]
    efnp1 = efn[sites_ip1]

    Nc = sys.Nc[sites_i]
    mu = sys.mu_e[sites_i]

    dv = dv + (np.abs(dv) < 1e-5)*1e-5

    jn = exp(-bl) * (exp(efnp1) - exp(efnp0)) / dl * \
         dv / (-exp(-vp0)*(1 - exp(dv)))

    return jn * Nc * mu

def get_jp(sys, efp, v, sites_i, sites_ip1, dl):
    bl = 0

    vp0 = v[sites_i]
    dv = vp0 - v[sites_ip1]
    efpp0= efp[sites_i]
    efpp1 = efp[sites_ip1]

    Nv = sys.Nv[sites_i]
    Eg = sys.Eg[sites_i]
    mu = sys.mu_h[sites_i]

    dv = dv + (np.abs(dv) < 1e-5)*1e-5

    jp = exp(-Eg + bl) * (exp(efpp1) - exp(efpp0)) / dl *\
         dv / (-exp(vp0)*(1 - exp(-dv)))

    return jp * Nv * mu

def get_jn_derivs(sys, efn, v, sites_i, sites_ip1, dl):
    bl = 0

    vp0 = v[sites_i]
    vp1 = v[sites_ip1]
    dv = vp0 - vp1
    efnp0= efn[sites_i]
    efnp1 = efn[sites_ip1]

    Nc = sys.Nc[sites_i]
    mu = sys.mu_e[sites_i]

    dv = dv + (np.abs(dv) < 1e-5)*1e-5
    # defn_i = 1./dl * exp(-bl+efnp0) * (-dv)\
    #          / (-exp(-vp0)*(1 - exp(dv)))
    #
    # defn_ip1 = -1./dl * exp(-bl+efnp1) * (-dv)\
    #            / (-exp(-vp0) * (1 - exp(dv)))
    #
    # dv_i = -1./dl * exp(-bl) * (exp(efnp1) - exp(efnp0))\
    #        * exp(-vp0)*(1 + dv - exp(dv))\
    #        / (exp(-2*vp0) * (exp(dv)-1)**2)
    #
    # dv_ip1 = -1./dl * exp(-bl) * (exp(efnp1) - exp(efnp0))\
    #         * exp(-vp1) * (1 - dv - exp(-dv))\
    #         / (exp(-2*vp1) * (1-exp(-dv))**2)

    ev0 = exp(-vp0)
    ep1 = exp(-bl+efnp1)
    ep0 = exp(-bl+efnp0)

    defn_i = 1./dl * ep0 * (-dv) / (-ev0*(1 - exp(dv)))

    defn_ip1 = -1./dl * ep1 * (-dv) / (-ev0 * (1 - exp(dv)))

    dv_i = -1./dl * (ep1 - ep0) * ev0*(1 + dv - exp(dv))\
           / (ev0**2 * (exp(dv)-1)**2)

    dv_ip1 = -1./dl * (ep1 - ep0) * exp(-vp1) * (1 - dv - exp(-dv))\
            / (exp(-2*vp1) * (1-exp(-dv))**2)

    return mu*Nc*defn_i, mu*Nc*defn_ip1, mu*Nc*dv_i, mu*Nc*dv_ip1   

def get_jp_derivs(sys, efp, v, sites_i, sites_ip1, dl):
    bl = 0

    vp0 = v[sites_i]
    vp1 = v[sites_ip1]
    dv = vp0 - vp1
    efpp0= efp[sites_i]
    efpp1 = efp[sites_ip1]

    Nv = sys.Nv[sites_i]
    Eg = sys.Eg[sites_i]
    mu = sys.mu_h[sites_i]

    dv = dv + (np.abs(dv) < 1e-5)*1e-5

    # defp_i = 1/dl * exp(bl-Eg+efpp0) * (-dv)\
    #          / (-exp(vp0) * (1 - exp(-dv)))
    #
    # defp_ip1 = 1/dl * exp(bl-Eg+efpp1) * (-dv)\
    #            / (exp(vp0) * (1 - exp(-dv)))
    #
    # dv_i = -1/dl * exp(bl-Eg) * (exp(efpp1) - exp(efpp0))\
    #        * exp(vp0) * (1-dv - exp(-dv))\
    #        / (exp(2*vp0)*((exp(-dv)-1)**2))
    #
    # dv_ip1 = -1/dl * exp(bl-Eg) * (exp(efpp1) - exp(efpp0))\
    #         * exp(vp1) * (1+dv - exp(dv))\
    #         / (exp(2*vp1)*((1-exp(dv))**2))

    ev0 = exp(vp0)
    ep1 = exp(bl+efpp1-Eg)
    ep0 = exp(bl+efpp0-Eg)

    defp_i = 1/dl * ep0 * (-dv) / (-ev0 * (1 - exp(-dv)))

    defp_ip1 = 1/dl * ep1 * (-dv) / (ev0 * (1 - exp(-dv)))

    dv_i = -1/dl * (ep1 - ep0) * ev0 * (1-dv - exp(-dv))\
           / (ev0**2*((exp(-dv)-1)**2))

    dv_ip1 = -1/dl * (ep1 - ep0) * exp(vp1) * (1+dv - exp(dv))\
            / (exp(2*vp1)*((1-exp(dv))**2))

    return mu*Nv*defp_i, mu*Nv*defp_ip1, mu*Nv*dv_i, mu*Nv*dv_ip1

def get_rr_derivs(sys, n, p, n1, p1, tau_e, tau_h, sites):
    ni = sys.ni[sites]

    defn = (n*p*(tau_h*(n+n1) + tau_e*(p+p1)) - (n*p-ni**2)*n*tau_h)\
         / (tau_h*(n+n1) + tau_e*(p+p1))**2
    defp = (n*p*(tau_h*(n+n1) + tau_e*(p+p1)) - (n*p-ni**2)*p*tau_e)\
         / (tau_h*(n+n1) + tau_e*(p+p1))**2
    dv = (n*p-ni**2) * (tau_e*p - tau_h*n) / (tau_h*(n+n1) + tau_e*(p+p1))**2

    return defn, defp, dv
