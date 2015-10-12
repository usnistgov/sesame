from numpy import exp
import numpy as np

def get_n(efn, v, params):
    bl = params.bl
    nC = params.nC
    return nC*exp(-bl+efn+v)

def get_p(efp, v, params):
    bl = params.bl
    eg = params.eg
    nV = params.nV
    return nV*exp(-eg+bl+efp-v)

def get_rr(n, p, n1, p1, S, params):
    ni = params.ni
    r = S * (n*p-ni**2)/(n+p+n1+p1)
    return r

def get_jn(efn, efnp1, v, vp1, dx, params):
    bl = params.bl
    nC = params.nC

    dv = v - vp1
    dv = dv + (np.abs(dv) < 1e-5)*1e-5

    jn = exp(-bl) * (exp(efnp1) - exp(efn)) / dx * \
         dv / (-exp(-v)*(1 - exp(dv)))

    return jn * nC

def get_jp(efp, efpp1, v, vp1, dx, params):
    bl = params.bl
    eg = params.eg
    nV = params.nV

    dv = v - vp1
    dv = dv + (np.abs(dv) < 1e-5)*1e-5

    jp = exp(-eg + bl) * (exp(efpp1) - exp(efp)) / dx *\
         dv / (-exp(v)*(1 - exp(-dv)))

    return jp * nV

def get_jn_derivs(efn_i, efn_ip1, v_i, v_ip1, dx_i, params):
    bl = params.bl
    nC = params.nC

    dv = v_i - v_ip1
    dv = dv + (np.abs(dv) < 1e-5)*1e-5

    defn_i = 1./dx_i * exp(-bl+efn_i) * (-dv)\
             / (-exp(-v_i)*(1 - exp(dv)))

    defn_ip1 = -1./dx_i * exp(-bl+efn_ip1) * (-dv)\
               / (-exp(-v_i) * (1 - exp(dv)))

    dv_i = -1./dx_i * exp(-bl) * (exp(efn_ip1) - exp(efn_i))\
           * exp(-v_i)*(1 + dv - exp(dv))\
           / (exp(-2*v_i) * (exp(dv)-1)**2)

    dv_ip1 = -1./dx_i * exp(-bl) * (exp(efn_ip1) - exp(efn_i))\
            * exp(-v_ip1) * (1 - dv - exp(-dv))\
            / (exp(-2*v_ip1) * (1-exp(-dv))**2)

    return nC*defn_i, nC*defn_ip1, nC*dv_i, nC*dv_ip1   

def get_jp_derivs(efp_i, efp_ip1, v_i, v_ip1, dx_i, params):
    bl = params.bl
    eg = params.eg
    nV = params.nV

    dv = v_i - v_ip1
    dv = dv + (np.abs(dv) < 1e-5)*1e-5

    defp_i = 1/dx_i * exp(bl-eg+efp_i) * (-dv)\
             / (-exp(v_i) * (1 - exp(-dv)))

    defp_ip1 = 1/dx_i * exp(bl-eg+efp_ip1) * (-dv)\
               / (exp(v_i) * (1 - exp(-dv)))

    dv_i = -1/dx_i * exp(bl-eg) * (exp(efp_ip1) - exp(efp_i))\
           * exp(v_i) * (1-dv - exp(-dv))\
           / (exp(2*v_i)*((exp(-dv)-1)**2))

    dv_ip1 = -1/dx_i * exp(bl-eg) * (exp(efp_ip1) - exp(efp_i))\
            * exp(v_ip1) * (1+dv - exp(dv))\
            / (exp(2*v_ip1)*((1-exp(dv))**2))

    return nV*defp_i, nV*defp_ip1, nV*dv_i, nV*dv_ip1

def get_rr_derivs(n, p, n1, p1, S, params):
    ni = params.ni

    defp = S * (n*p*(n1+p1+n+p) - (n*p-ni**2)*p) / (n1+p1+n+p)**2
    defn = S * (n*p*(n1+p1+n+p) - (n*p-ni**2)*n) / (n1+p1+n+p)**2
    dv = S * (n*p-ni**2) * (p-n) / (n1+p1+n+p)**2

    return defp, defn, dv
