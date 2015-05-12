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

def get_rr(n, p, n1, p1, S, params):
    ni = params.ni
    r = S * (n*p-ni**2)/(n+p+n1+p1)
    return r

def get_jn(efn, efnp1, v, vp1, dx, params):
    bl = params.bl
    nC = params.nC

    h = 1e-30

    jn = exp(-bl) * (exp(efnp1) - exp(efn)) / dx * \
         (v - vp1 + h) / (-exp(-v)*(1 - exp(-vp1+v) + h))
    return jn * nC

def get_jp(efp, efpp1, v, vp1, dx, params):
    bl = params.bl
    eg = params.eg
    nV = params.nV

    h = 1e-30
    
    jp = exp(-eg + bl) * (exp(efpp1) - exp(efp)) / dx *\
         (v - vp1 + h) / (-exp(v)*(1 - exp(vp1-v)  + h))

    return jp * nV

def get_jn_derivs(efn_i, efn_ip1, v_i, v_ip1, dx_i, params):
    bl = params.bl
    nC = params.nC

    h = 1e-30

    defn_i = 1./dx_i * exp(-bl+efn_i) * (v_ip1 - v_i + h)\
             / (-exp(-v_i)*(1 - exp(-v_ip1+v_i) + h))

    defn_ip1 = -1./dx_i * exp(-bl+efn_ip1) * (v_ip1 - v_i + h)\
               / (-exp(-v_i) * (1 - exp(-v_ip1+v_i) + h))

    dv_i = -1./dx_i * exp(-bl) * (exp(efn_ip1) - exp(efn_i))\
           * exp(-v_i)*(1+v_i-v_ip1 - exp(-v_ip1+v_i) - h/2)\
           / (exp(-2*v_i) * ((exp(-v_ip1-v_i)-1)**2 + h))

    dv_ip1 = -1./dx_i * exp(-bl) * (exp(efn_ip1) - exp(efn_i))\
            * exp(-v_ip1) * (1+v_ip1-v_i - exp(-v_i+v_ip1) - h/2)\
            / (exp(-2*v_ip1) * ((1-exp(v_ip1-v_i))**2 + h))

    return nC*defn_i, nC*defn_ip1, nC*dv_i, nC*dv_ip1   

def get_jp_derivs(efp_i, efp_ip1, v_i, v_ip1, dx_i, params):
    bl = params.bl
    eg = params.eg
    nV = params.nV

    h = 1e-30

    defp_i = 1/dx_i * exp(bl-eg+efp_i) * (v_ip1 - v_i + h)\
             / (-exp(v_i) * (1 - exp(v_ip1-v_i) + h))

    defp_ip1 = 1/dx_i * exp(bl-eg+efp_ip1) * (v_ip1 - v_i + h)\
               / (exp(v_i) * (1 - exp(v_ip1-v_i) + h))

    dv_i = -1/dx_i * exp(bl-eg) * (exp(efp_ip1) - exp(efp_i))\
           * exp(v_i) * (1+v_ip1-v_i - exp(v_ip1-v_i) - h/2)\
           / (exp(2*v_i)*((exp(v_ip1-v_i)-1)**2 + h))

    dv_ip1 = -1/dx_i * exp(bl-eg) * (exp(efp_ip1) - exp(efp_i))\
            * exp(v_ip1) * (1-v_ip1+v_i - exp(v_i-v_ip1) - h/2)\
            / (exp(2*v_ip1)*((1-exp(-v_ip1+v_i))**2 + h))

    return nV*defp_i, nV*defp_ip1, nV*dv_i, nV*dv_ip1

def get_rr_derivs(n, p, n1, p1, S, params):
    ni = params.ni

    defp = S * (n*p*(n1+p1+n+p) - (n*p-ni**2)*p) / (n1+p1+n+p)**2
    defn = S * (n*p*(n1+p1+n+p) - (n*p-ni**2)*n) / (n1+p1+n+p)**2
    dv = S * (n*p-ni**2) * (p-n) / (n1+p1+n+p)**2

    return defp, defn, dv
