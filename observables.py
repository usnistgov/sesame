from numpy import exp

def get_n(efn, v, params):
    bl = params.bl
    nC = params.nC
    return nC*exp(-bl+efn+v)

def get_p(efp, v, params):
    bl = params.bl
    eg = params.eg
    nV = params.nV
    return nV*exp(-eg+bl+efp-v)

def get_rr(efn, efp, v, tau, params):
    eg = params.eg
    n = get_n(efn, v, params)
    p = get_p(efp, v, params)
    ni = exp(-eg/2.)

    r = (n*p-ni**2)/(n+p+2*ni) / tau
    return r

def get_rrGB(efn, efp, v, tau, params):
    eg = params.eg
    n = get_n(efn, v, params)
    p = get_p(efp, v, params)
    ni = exp(-eg/2.)
    nGB = params.nGB
    pGB = params.pGB

    r = (n*p-ni**2)/(n+p+nGB+pGB) / tau
    return r

def get_jn(efn, efnp1, v, vp1, dx, params):
    bl = params.bl
    nC = params.nC

    if abs(v-vp1) > 1e-8:
        jn = (v - vp1)/dx * exp(-bl) * (exp(efnp1) - exp(efn))\
             / (exp(-vp1) - exp(-v))
    else:
        jn = exp(v)/dx * exp(-bl) * (exp(efnp1) - exp(efn))

    return jn * nC

def get_jp(efp, efpp1, v, vp1, dx, params):
    bl = params.bl
    eg = params.eg
    nV = params.nV

    if abs(v-vp1) > 1e-8:
        jp = - (vp1 - v)/dx * exp(-eg + bl) * (exp(efpp1) - exp(efp))\
             / (exp(vp1) - exp(v))
    else:
        jp = - exp(-v)/dx * exp(-eg + bl) * (exp(efpp1) - exp(efp))

    return jp * nV

def get_jn_derivs(efn_i, efn_ip1, v_i, v_ip1, dx_i, params):
    bl = params.bl
    nC = params.nC

    if abs(v_i-v_ip1) > 1e-8:
        defn_i = 1./dx_i * exp(-bl+efn_i) * (v_ip1 - v_i)\
                 / (exp(-v_ip1) - exp(-v_i))
        defn_ip1 = -1./dx_i * exp(-bl+efn_ip1) * (v_ip1 - v_i)\
                   / (exp(-v_ip1) - exp(-v_i))
        dv_i = -1./dx_i * exp(-bl) * (exp(efn_ip1) - exp(efn_i))\
               * (exp(-v_i) * (1+v_i-v_ip1) - exp(-v_ip1))\
               / (exp(-v_ip1) - exp(-v_i))**2 
        dv_ip1 = -1./dx_i * exp(-bl) * (exp(efn_ip1) - exp(efn_i))\
                * (exp(-v_ip1) * (1+v_ip1-v_i) - exp(-v_i))\
                / (exp(-v_ip1) - exp(-v_i))**2 
    else:
        defn_i = -1./dx_i * exp(-bl+efn_i) * exp(v_i)
        defn_ip1 = 1./dx_i * exp(-bl+efn_ip1) * exp(v_i)
        dv_i = 1./dx_i * exp(-bl) * (exp(efn_ip1) - exp(efn_i))\
               * exp(v_i) / 2.
        dv_ip1 = dv_i

    return nC*defn_i, nC*defn_ip1, nC*dv_i, nC*dv_ip1   

def get_jp_derivs(efp_i, efp_ip1, v_i, v_ip1, dx_i, params):
    bl = params.bl
    eg = params.eg
    nV = params.nV

    if abs(v_i-v_ip1) > 1e-8:
        defp_i = 1/dx_i * exp(bl-eg+efp_i) * (v_ip1 - v_i)\
                 / (exp(v_ip1) - exp(v_i))
        defp_ip1 = -1/dx_i * exp(bl-eg+efp_ip1) * (v_ip1 - v_i)\
                   / (exp(v_ip1) - exp(v_i))
        dv_i = -1/dx_i * exp(bl-eg) * (exp(efp_ip1) - exp(efp_i))\
               * (exp(v_i) * (1+v_ip1-v_i) - exp(v_ip1))\
               / (exp(v_ip1) - exp(v_i))**2
        dv_ip1 = -1/dx_i * exp(bl-eg) * (exp(efp_ip1) - exp(efp_i))\
                * (exp(v_ip1) * (1-v_ip1+v_i) - exp(v_i))\
                / (exp(v_ip1)-exp(v_i))**2
        
    else:
        defp_i = 1/dx_i * exp(bl-eg+efp_i) * exp(-v_i)
        defp_ip1 = -1/dx_i * exp(bl-eg+efp_ip1) * exp(-v_i)
        dv_i = 1/dx_i * exp(bl-eg) * (exp(efp_ip1) - exp(efp_i))\
               * exp(-v_i) / 2.
        dv_ip1 = dv_i
        
    return nV*defp_i, nV*defp_ip1, nV*dv_i, nV*dv_ip1

def get_rr_derivs(efn_i, efp_i, v_i, tau, params):
    eg = params.eg
    n = get_n(efn_i, v_i, params)
    p = get_p(efp_i, v_i, params)
    ni = exp(-eg/2.)

    defp_i = 1/tau * (n*p*(2*ni+n+p) - (n*p-ni**2)*p) / (2*ni+n+p)**2
    defn_i = 1/tau * (n*p*(2*ni+n+p) - (n*p-ni**2)*n) / (2*ni+n+p)**2
    dv_i = 1/tau * (n*p-ni**2) * (p-n) / (2*ni+n+p)**2

    return defp_i, defn_i, dv_i

def get_rrGB_derivs(efn, efp, v, tau, params):
    eg = params.eg
    n = get_n(efn, v, params)
    p = get_p(efp, v, params)
    ni = exp(-eg/2.)
    nGB = params.nGB
    pGB = params.pGB

    defp_i = 1/tau * (n*p*(n+p+nGB+pGB) - (n*p-ni**2)*p) / (nGB+pGB+n+p)**2
    defn_i = 1/tau * (n*p*(n+p+nGB+pGB) - (n*p-ni**2)*n) / (nGB+pGB+n+p)**2
    dv_i = 1/tau * (n*p-ni**2) * (p-n) / (n+p+nGB+pGB)**2

    return defp_i, defn_i, dv_i
