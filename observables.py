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

def get_rr(efn, efp, v, n1, p1, S, params):
    ni = params.ni
    n = get_n(efn, v, params)
    p = get_p(efp, v, params)

    r = S * (n*p-ni**2)/(n+p+n1+p1)
    return r

def get_jn(efn_s, efn_sp1, v_s, v_sp1, v_smN, v_spN, dx, params):
    bl = params.bl
    nC = params.nC

    jn = (v_s - v_sp1) / dx * exp(-bl) * (exp(efn_sp1) - exp(efn_s))\
         * (exp(-2*v_s-v_spN) - exp(-v_s-v_spN-v_smN) - exp(-3*v_s)\
            + exp(-2*v_s-v_smN) - exp(-v_s-v_sm1-v_spN) + exp(-v_sm1-v_spN-v_smN))

    return jn * nC

def get_jp(efp_s, efp_sp1, v_s, v_sp1, v_smN, v_spN, dx, params):
    bl = params.bl
    eg = params.eg
    nV = params.nV

    jp = (v_s - v_sp1)/dx * exp(-eg + bl) * (exp(efp_sp1) - exp(efp_s))\
         * (exp(2*v_s+v_spN) - exp(v_s+v_spN+v_smN) - exp(3*v_s)\
            + exp(2*v_s+v_smN) - exp(v_s+v_sm1+v_spN) + exp(v_sm1+v_spN+v_smN))

    return jp * nV

def get_jn_derivs(efn_s, efn_sp1, v_sm1, v_s, v_sp1, v_smN, v_spN, dx, params):
    bl = params.bl
    nC = params.nC

    a = exp(-2*v_s-v_spN) - exp(-v_s-v_spN-v_smN) - exp(-3*v_s)\
        + exp(-2*v_s-v_smN) - exp(-v_s-v_sm1-v_spN) + exp(-v_sm1-v_spN-v_smN)\
        + exp(-2*v_s-v_sm1) - exp(-v_s-v_sm1-v_smN)
    b = exp(-bl) * (exp(efn_sp1) - exp(efn_s))
    c = v_s - v_sp1
    d = b * c

    defn_s = exp(-bl+efn_s) * c * p / dx

    defn_sp1 = exp(-bl+efn_sp1) * c * p / dx

    dv_smN = d * (exp(-v_s-v_spN-v_smN) - exp(-2*v_s-v_smN) -\
             exp(-v_sm1-v_spN-v_smN) + exp(-v_s-v_sm1-v_smN)) / dx

    dv_spN = d * (-exp(-2*v_s-v_spN) + exp(-v_s-v_spN-v_smN) +\
             exp(-v_s-v_sm1-v_spN) - exp(-v_sm1-vspN-v_smN)) / dx

    dv_s = b * p + d * (-2*exp(-2*v_s-v_spN) + exp(-v_s-v_spN-v_smN) +\
           3*exp(-3*v_s) - 2*exp(-2*v_s-v_smN) + exp(-v_s-v_sm1-v_spN) -\
           2*exp(-2*v_s-v_sm1) + exp(-v_s-v_sm1-v_smN)) / dx

    dv_sm1 = d * (exp(-v_s-v_sm1-v_spN) - exp(-v_sm1-v_spN-v_smN) -\
             exp(-2*v_s-v_sm1) + exp(-v_s-v_sm1-v_smN)) / dx

    dv_sp1 = - b * p

    return nC*defn_s, nC*defn_sp1, nC*dv_s, nC*dv_sp1, nC*dv_smN, nC*dv_spN

def get_jp_derivs(efp_s, efp_sp1, v_sm1, v_s, v_sp1, v_smN, v_spN, dx, params):
    bl = params.bl
    eg = params.eg
    nV = params.nV

    a = exp(2*v_s+v_spN) - exp(v_s+v_spN+v_smN) - exp(3*v_s)\
        + exp(2*v_s+v_smN) - exp(v_s+v_sm1+v_spN) + exp(v_sm1+v_spN+v_smN)\
        + exp(2*v_s+v_sm1) - exp(v_s+v_sm1+v_smN)
    b = exp(bl-eg) * (exp(efn_sp1) - exp(efn_s))
    c = v_s - v_sp1
    d = b * c

    defn_s = exp(bl-eg+efn_s) * c * p / dx

    defn_sp1 = exp(bl-eg+efn_sp1) * c * p / dx

    dv_smN = d * (-exp(v_s+v_spN+v_smN) + exp(2*v_s+v_smN) +\
             exp(v_sm1+v_spN+v_smN) + exp(v_s+v_sm1+v_smN)) / dx

    dv_spN = d * (exp(2*v_s+v_spN) - exp(v_s+v_spN+v_smN) -\
             exp(v_s+v_sm1+v_spN) + exp(v_sm1+vspN+v_smN)) / dx

    dv_s = b * p + d * (2*exp(2*v_s+v_spN) - exp(v_s+v_spN+v_smN) -\
           3*exp(3*v_s) + 2*exp(2*v_s+v_smN) - exp(v_s+v_sm1+v_spN) -\
           2*exp(2*v_s+v_sm1) - exp(v_s+v_sm1+v_smN)) / dx

    dv_sm1 = d * (-exp(v_s+v_sm1+v_spN) + exp(v_sm1+v_spN+v_smN) +\
             exp(2*v_s+v_sm1) - exp(v_s+v_sm1+v_smN)) / dx

    dv_sp1 = - b * p

    return nV*defn_s, nV*defn_sp1, nV*dv_s, nV*dv_sp1, nV*dv_smN, nV*dv_spN

def get_rr(n, p, n1, p1, S, params):
    ni = params.ni
    r = S * (n*p-ni**2)/(n+p+n1+p1)
    return r


def get_rr_derivs(efn_i, efp_i, v_i, n1, p1, S, params):
    ni = params.ni
    n = get_n(efn_i, v_i, params)
    p = get_p(efp_i, v_i, params)

    defp_i = S * (n*p*(n1+p1+n+p) - (n*p-ni**2)*p) / (n1+p1+n+p)**2
    defn_i = S * (n*p*(n1+p1+n+p) - (n*p-ni**2)*n) / (n1+p1+n+p)**2
    dv_i = S * (n*p-ni**2) * (p-n) / (n1+p1+n+p)**2

    return defp_i, defn_i, dv_i
