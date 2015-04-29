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

def get_jvn_s(efn_s, efn_sp1, v_sm1,  v_s, v_sp1, v_smN, v_spN, dx, params):
    bl = params.bl
    nC = params.nC

    jvn_s = (v_s - v_sp1) / dx * exp(-bl) * (exp(efn_sp1) - exp(efn_s))\
         * (exp(-2*v_s-v_spN) - exp(-v_s-v_spN-v_smN) - exp(-3*v_s)\
            + exp(-2*v_s-v_smN) - exp(-v_s-v_sm1-v_spN) + exp(-v_sm1-v_spN-v_smN))
    return nC * jvn_s

def get_jvn_sm1(efn_sm1, efn_s, v_sm1, v_s, v_sp1, v_smN, v_spN, dx, params):
    bl = params.bl
    nC = params.nC

    jvn_sm1 = (v_sm1 - v_s) / dx * exp(-bl) * (exp(efn_s) - exp(efn_sm1)) *\
              (exp(-v_sp1)-exp(-v_s)) * (exp(-v_spN) - exp(-v_s)) *\
              (exp(-v_s)-exp(-v_smN))
    return nC * jvn_sm1

def get_jvp_s(efp_s, efp_sp1, v_sm1, v_s, v_sp1, v_smN, v_spN, dx, params):
    bl = params.bl
    eg = params.eg
    nV = params.nV

    jp_s = (v_s - v_sp1)/dx * exp(-eg + bl) * (exp(efp_sp1) - exp(efp_s))\
         * (exp(2*v_s+v_spN) - exp(v_s+v_spN+v_smN) - exp(3*v_s)\
            + exp(2*v_s+v_smN) - exp(v_s+v_sm1+v_spN) + exp(v_sm1+v_spN+v_smN))
    return nV * jp_s

def get_jvp_sm1(efp_sm1, efp_s, v_sm1, v_s, v_sp1, v_smN, v_spN, dx, params):
    bl = params.bl
    eg = params.eg
    nV = params.nV

    jvp_sm1 = (v_sm1 - v_s) / dx * exp(-eg + bl) * (exp(efp_s) - exp(efp_sm1)) *\
              (exp(v_sp1)-exp(v_s)) * (exp(v_spN) - exp(v_s)) *\
              (exp(v_s)-exp(v_smN))
    return nV * jvp_sm1

def get_jvn_s_derivs(efn_s, efn_sp1, v_sm1, v_s, v_sp1, v_smN, v_spN, dx, params):
    bl = params.bl
    nC = params.nC

    a = exp(-2*v_s-v_spN) - exp(-v_s-v_spN-v_smN) - exp(-3*v_s)\
        + exp(-2*v_s-v_smN) - exp(-v_s-v_sm1-v_spN) + exp(-v_sm1-v_spN-v_smN)\
        + exp(-2*v_s-v_sm1) - exp(-v_s-v_sm1-v_smN)
    b = exp(-bl) * (exp(efn_sp1) - exp(efn_s))
    c = v_s - v_sp1
    d = b * c

    defn_s = -exp(-bl+efn_s) * c * a / dx

    defn_sp1 = exp(-bl+efn_sp1) * c * a / dx

    dv_smN = d * (exp(-v_s-v_spN-v_smN) - exp(-2*v_s-v_smN) -\
             exp(-v_sm1-v_spN-v_smN) + exp(-v_s-v_sm1-v_smN)) / dx

    dv_spN = d * (-exp(-2*v_s-v_spN) + exp(-v_s-v_spN-v_smN) +\
             exp(-v_s-v_sm1-v_spN) - exp(-v_sm1-v_spN-v_smN)) / dx

    dv_s = b * a + d * (-2*exp(-2*v_s-v_spN) + exp(-v_s-v_spN-v_smN) +\
           3*exp(-3*v_s) - 2*exp(-2*v_s-v_smN) + exp(-v_s-v_sm1-v_spN) -\
           2*exp(-2*v_s-v_sm1) + exp(-v_s-v_sm1-v_smN)) / dx

    dv_sm1 = d * (exp(-v_s-v_sm1-v_spN) - exp(-v_sm1-v_spN-v_smN) -\
             exp(-2*v_s-v_sm1) + exp(-v_s-v_sm1-v_smN)) / dx

    dv_sp1 = - b * a

    return nC*defn_s, nC*defn_sp1, nC*dv_sm1, nC*dv_s, nC*dv_sp1, nC*dv_smN, nC*dv_spN

def get_jvn_sm1_derivs(efn_sm1, efn_s, v_sm1, v_s, v_sp1, v_smN, v_spN, dx, params):
    # to get the same thing in the y-direction, change 1 by N and N by 1
    bl = params.bl
    nC = params.nC

    a = exp(-v_sp1) - exp(-v_s)
    b = exp(-v_spN) - exp(-v_s)
    c = exp(-v_s) - exp(-v_smN)
    d = (v_s - v_sm1) * exp(-bl) * (exp(efn_s) - exp(efn_sm1))

    defn_s = exp(-bl+efn_s) * (v_s - v_sm1) * a * b * c
    defn_sm1 = -exp(-bl+efn_sm1) * (v_s - v_sm1) * a * b * c

    dv_sm1 = - exp(-bl) * (exp(efn_s) - exp(efn_sm1)) * a * b * c
    dv_sp1 = - d * exp(-v_sp1) * b * c
    dv_spN = - d * exp(-v_spN) * a * c
    dv_smN = d * exp(-v_smN) * a * b
    dv_s   = exp(-bl) * (exp(efn_s) - exp(efn_sm1)) * \
             (a*b*c + (v_s-v_sm1)*exp(-v_s) * (b*c + a*c - a*b))

    return nC*defn_sm1, nC*defn_s, nC*dv_sm1, nC*dv_s, nC*dv_sp1, nC*dv_smN, nC*dv_spN


def get_jvp_s_derivs(efp_s, efp_sp1, v_sm1, v_s, v_sp1, v_smN, v_spN, dx, params):
    bl = params.bl
    eg = params.eg
    nV = params.nV

    a = exp(2*v_s+v_spN) - exp(v_s+v_spN+v_smN) - exp(3*v_s)\
        + exp(2*v_s+v_smN) - exp(v_s+v_sm1+v_spN) + exp(v_sm1+v_spN+v_smN)\
        + exp(2*v_s+v_sm1) - exp(v_s+v_sm1+v_smN)
    b = exp(bl-eg) * (exp(efp_sp1) - exp(efp_s))
    c = v_s - v_sp1
    d = b * c

    defp_s = exp(bl-eg+efp_s) * c * a / dx

    defp_sp1 = exp(bl-eg+efp_sp1) * c * a / dx

    dv_smN = d * (-exp(v_s+v_spN+v_smN) + exp(2*v_s+v_smN) +\
             exp(v_sm1+v_spN+v_smN) + exp(v_s+v_sm1+v_smN)) / dx

    dv_spN = d * (exp(2*v_s+v_spN) - exp(v_s+v_spN+v_smN) -\
             exp(v_s+v_sm1+v_spN) + exp(v_sm1+v_spN+v_smN)) / dx

    dv_s = b * a + d * (2*exp(2*v_s+v_spN) - exp(v_s+v_spN+v_smN) -\
           3*exp(3*v_s) + 2*exp(2*v_s+v_smN) - exp(v_s+v_sm1+v_spN) -\
           2*exp(2*v_s+v_sm1) - exp(v_s+v_sm1+v_smN)) / dx

    dv_sm1 = d * (-exp(v_s+v_sm1+v_spN) + exp(v_sm1+v_spN+v_smN) +\
             exp(2*v_s+v_sm1) - exp(v_s+v_sm1+v_smN)) / dx

    dv_sp1 = - b * a

    return nV*defp_s, nV*defp_sp1, nV*dv_sm1, nV*dv_s, nV*dv_sp1, nV*dv_smN, nV*dv_spN

def get_jvp_sm1_derivs(efp_sm1, efp_s, v_sm1, v_s, v_sp1, v_smN, v_spN, dx, params):
    # to get the same thing in the y-direction, change 1 by N and N by 1
    bl = params.bl
    eg = params.eg
    nV = params.nV

    a = exp(v_sp1) - exp(v_s)
    b = exp(v_spN) - exp(v_s)
    c = exp(v_s) - exp(v_smN)
    d = exp(bl-eg) * (v_s - v_sm1) * (exp(efp_s) - exp(efp_sm1))

    defp_s = exp(bl-eg+efp_s) * (v_s - v_sm1) * a * b * c
    defp_sm1 = -exp(bl-eg+efp_sm1) * (v_s - v_sm1) * a * b * c

    dv_sm1 = - exp(bl-eg) * (exp(efp_s) - exp(efp_sm1)) * a * b * c
    dv_sp1 = d * exp(v_sp1) * b * c
    dv_spN = d * exp(v_spN) * a * c
    dv_smN = - d * exp(v_smN) * a * b
    dv_s   = exp(bl-eg) * (exp(efp_s) - exp(efp_sm1)) *\
            (a*b*c + (v_s-v_sm1)*exp(v_s) * (-b*c - a*c + a*b))

    return nV*defp_sm1, nV*defp_s, nV*dv_sm1, nV*dv_s, nV*dv_sp1, nV*dv_smN, nV*dv_spN


def get_rr(n, p, n1, p1, S, params):
    ni = params.ni
    r = S * (n*p-ni**2)/(n+p+n1+p1)
    return r

def get_uvn(n, p, v_smN, v_sm1, v_s, v_sp1, v_spN, g, S, SGB, params):
    ni = params.ni
    n1 = params.n1
    p1 = params.p1
    nGB = params.nGB
    pGB = params.pGB

    a = exp(-v_sp1) - exp(-v_s)
    b = exp(-v_s) - exp(-v_sm1)
    c = exp(-v_spN) - exp(-v_s)
    d = exp(-v_s) - exp(-v_smN)

    uvn = a * b * c * d * (g - (n*p - ni**2) * (S / (n+p+n1+p1) + SGB / (n+p+nGB+pGB)))
    return uvn

def get_uvp(n, p, v_smN, v_sm1, v_s, v_sp1, v_spN, g, S, SGB, params):
    ni = params.ni
    n1 = params.n1
    p1 = params.p1
    nGB = params.nGB
    pGB = params.pGB

    a = exp(v_sp1) - exp(v_s)
    b = exp(v_s) - exp(v_sm1)
    c = exp(v_spN) - exp(v_s)
    d = exp(v_s) - exp(v_smN)

    uvp = a * b * c * d * (g - (n*p - ni**2) * (S / (n+p+n1+p1) + SGB / (n+p+nGB+pGB)))
    return uvp

def get_uvn_derivs(n, p, v_smN, v_sm1, v_s, v_sp1, v_spN, g, S, SGB, params):
    ni = params.ni
    n1 = params.n1
    p1 = params.p1
    nGB = params.nGB
    pGB = params.pGB

    a = exp(-v_sp1) - exp(-v_s)
    b = exp(-v_s) - exp(-v_sm1)
    c = exp(-v_spN) - exp(-v_s)
    d = exp(-v_s) - exp(-v_smN)
    e = a * b * c * d
    np = n * p
    npni = np - ni**2
    na = n + p + n1 + p1
    nb = n + p + nGB + pGB
    r = S * npni / na
    rGB = SGB * npni / nb

    defn_s = - e * (S * (np*na - npni*n) / na**2 + SGB * (np*nb - npni*n) / nb**2)

    defp_s = - e * (S * (np*na - npni*p) / na**2 + SGB * (np*nb - npni*p) / nb**2)

    dv_smN = exp(-v_smN) * a * b * c * (g - r - rGB)

    dv_sm1 = exp(-v_sm1) * a * c * d * (g - r - rGB)

    dv_s = - e * npni * (p-n) * (S / na**2 + SGB / nb**2) + (g - r - rGB) *\
            exp(-v_s) * (b*c*d + a*b*d - a*b*c - a*c*d)
    
    dv_sp1 = -exp(-v_sp1) * b * c * d * (g - r - rGB)

    dv_spN = -exp(-v_spN) * a * b * d * (g - r - rGB)

    return defn_s, defp_s, dv_smN, dv_sm1, dv_s, dv_sp1, dv_spN

def get_uvp_derivs(n, p, v_smN, v_sm1, v_s, v_sp1, v_spN, g, S, SGB, params):
    ni = params.ni
    n1 = params.n1
    p1 = params.p1
    nGB = params.nGB
    pGB = params.pGB

    a = exp(v_sp1) - exp(v_s)
    b = exp(v_s) - exp(v_sm1)
    c = exp(v_spN) - exp(v_s)
    d = exp(v_s) - exp(v_smN)
    e = a * b * c * d
    np = n * p
    npni = np - ni**2
    na = n + p + n1 + p1
    nb = n + p + nGB + pGB
    r = S * npni / na
    rGB = SGB * npni / nb

    defn_s = - e * (S * (np*na - npni*n) / na**2 + SGB * (np*nb - npni*n) / nb**2)

    defp_s = - e * (S * (np*na - npni*p) / na**2 + SGB * (np*nb - npni*p) / nb**2)

    dv_smN = -exp(v_smN) * a * b * c * (g - r - rGB)

    dv_sm1 = -exp(v_sm1) * a * c * d * (g - r - rGB)

    dv_s = - e * npni * (p-n) * (S / na**2 + SGB / nb**2) + (g - r - rGB) *\
            exp(v_s) * (-b*c*d - a*b*d + a*b*c + a*c*d)

    dv_sp1 = exp(v_sp1) * b * c * d * (g - r - rGB)

    dv_spN = exp(v_spN) * a * b * d * (g - r - rGB)

    return defn_s, defp_s, dv_smN, dv_sm1, dv_s, dv_sp1, dv_spN

def get_jvbnx_sm1_derivs(efn_sm1, efn_s, v_sm1, v_s, v_smN, v_spN, dx, params):
    bl = params.bl
    nC = params.nC

    a = exp(-bl) * (exp(efn_s) - exp(efn_sm1))

    defn_s = (v_sm1 - v_s) / dx * exp(-bl+efn_s) * (exp(-v_spN) - exp(-v_s)) *\
             (exp(-v_s) - exp(-v_smN))

    defn_sm1 = - (v_sm1 - v_s) / dx * exp(-bl+efn_sm1) * (exp(-v_spN) - exp(-v_s)) *\
               (exp(-v_s) - exp(-v_smN))

    dv_s = a / dx * (- (exp(-v_spN) - exp(-v_s)) * (exp(-v_s) - exp(-v_smN)) \
                     + exp(-v_s) * (exp(-v_s) - exp(-v_smN)) * (v_sm1 - v_s) \
                     - exp(-v_s) * (exp(-v_spN) - exp(-v_s) * (v_sm1 - v_s)))
    
    dv_sm1 = a / dx

    dv_spN = - a / dx * (v_sm1 - v_s) * exp(-v_spN) * (exp(-v_s) - exp(-v_smN))

    dv_smN = a / dx * (v_sm1 - v_s) * exp(-v_smN) * (exp(-v_spN) - exp(-v_s))

    return nC*defn_sm1, nC*defn_s, nC*dv_smN, nC*dv_sm1, nC*dv_s, nC*dv_spN

def get_jvbny_s_derivs(efn_s, efn_spN, v_sm1, v_s, v_smN, v_spN, dy, params):
    bl = params.bl
    nC = params.nC

    a = exp(-bl) * (exp(efn_spN) - exp(efn_s))

    defn_s = - exp(-bl+efn_s) * (v_s-v_spN) * (exp(-v_s) - exp(-v_sm1)) *\
             (exp(-v_s) - exp(-v_smN)) / dy

    defn_spN = exp(-bl+efn_spN) * (v_s-v_spN) * (exp(-v_s) - exp(-v_sm1)) *\
             (exp(-v_s) - exp(-v_smN)) / dy

    dv_s = a / dy * ((exp(-v_s) - exp(-v_sm1)) * (exp(-v_s) - exp(-v_smN)) -\
    (v_s-v_spN) * exp(-v_s) * ((exp(-v_s) - exp(-v_smN)) + (exp(-v_s) - exp(-v_sm1))))

    dv_sm1 = a / dy * (v_s-v_spN) * exp(-v_sm1) * (exp(-v_s) - exp(-v_smN))

    dv_spN = - a / dy * (exp(-v_s) - exp(-v_sm1)) * (exp(-v_s) - exp(-v_smN))

    dv_smN = a / dy * (v_s-v_spN) * exp(-v_spN) * (exp(-v_s) - exp(-v_sm1))

    return nC*defn_s, nC*defn_spN, nC*dv_smN, nC*dv_sm1, nC*dv_s, nC*dv_spN
    
def get_jvbny_smN_derivs(efn_smN, efn_s, v_sm1, v_s, v_smN, v_spN, dy, params):
    bl = params.bl
    nC = params.nC

    a = exp(-bl) * (exp(efn_s) - exp(efn_smN))

    defn_s = exp(-bl+efn_s) * (v_smN-v_s) * (exp(-v_s) - exp(-v_sm1)) *\
             (exp(-v_spN) - exp(-v_s)) / dy

    defn_smN = - exp(-bl+efn_smN) * (v_smN-v_s) * (exp(-v_s) - exp(-v_sm1)) *\
               (exp(-v_spN) - exp(-v_s)) / dy

    dv_s = a / dy * (-(exp(-v_s) - exp(-v_sm1)) * (exp(-v_spN) - exp(-v_s)) +
    exp(-v_s)*(v_smN-v_s) * (-(exp(-v_spN) - exp(-v_s)) + (exp(-v_s) - exp(-v_sm1))))

    dv_sm1 = a / dy * (v_smN-v_s) * (exp(-v_spN) - exp(-v_s)) * exp(-v_sm1)

    dv_spN = - a / dy * (v_smN-v_s) * (exp(-v_s) - exp(-v_sm1)) * exp(-v_spN)

    dv_smN = a / dy * (exp(-v_spN) - exp(-v_s)) * (exp(-v_s) - exp(-v_sm1))

    return nC*defn_smN, nC*defn_s, nC*dv_smN, nC*dv_sm1, nC*dv_s, nC*dv_spN


def get_uvbn_derivs(n, p, efn_s, v_sm1, v_s, v_smN, v_spN, g, S, SGB, params):
    ni = params.ni
    n1 = params.n1
    p1 = params.p1
    nGB = params.nGB
    pGB = params.pGB

    r = S * (n*p - ni**2) / (n+p+n1+p1)
    rGB = SGB * (n*p - ni**2) / (n+p+nGB+pGB)

    defn = -(S * (n*p*(n1+p1+n+p) - (n*p-ni**2)*n) / (n1+p1+n+p)**2 +\
            SGB * (n*p*(nGB+pGB+n+p) - (n*p-ni**2)*n) / (nGB+pGB+n+p)**2) *\
            (exp(-v_s)-exp(-v_sm1)) * (exp(-v_spN) - exp(-v_s)) * (exp(-v_s) - exp(-v_smN))

    defp = -(S * (n*p*(n1+p1+n+p) - (n*p-ni**2)*p) / (n1+p1+n+p)**2 +\
            SGB * (n*p*(nGB+pGB+n+p) - (n*p-ni**2)*p) / (nGB+pGB+n+p)**2) *\
            (exp(-v_s)-exp(-v_sm1)) * (exp(-v_spN) - exp(-v_s)) * (exp(-v_s) - exp(-v_smN))

    dv_s = -(S * (n*p-ni**2) * (p-n) / (n1+p1+n+p)**2 +\
            SGB * (n*p-ni**2) * (p-n) / (nGB+pGB+n+p)**2) *\
           (exp(-v_s)-exp(-v_sm1)) * (exp(-v_spN) - exp(-v_s)) * (exp(-v_s) - exp(-v_smN))\
           + (g - r - rGB) * exp(-v_s) * \
           ((exp(-v_spN) - exp(-v_s)) * (exp(-v_s) - exp(-v_smN))\
           - (exp(-v_s)-exp(-v_sm1)) * (exp(-v_s) - exp(-v_smN))\
           + (exp(-v_s)-exp(-v_sm1)) * (exp(-v_spN) - exp(-v_s)))
    
    dv_sm1 = -(g - r - rGB) * exp(-v_sm1) * (exp(-v_spN) - exp(-v_s)) *\
              (exp(-v_s) - exp(-v_smN))

    dv_spN = (g - r - rGB) * exp(-v_spN) * (exp(-v_s) - exp(-v_sm1)) *\
             (exp(-v_s) - exp(-v_smN))

    dv_smN = -(g - r - rGB) * exp(-v_smN) * (exp(-v_s) - exp(-v_sm1)) *\
              (exp(-v_spN) - exp(-v_s))

    return defn, defp, dv_smN, dv_sm1, dv_s, dv_spN

def get_jvbpx_sm1_derivs(efp_sm1, efp_s, v_sm1, v_s, v_smN, v_spN, dx, params):
    bl = params.bl
    nV = params.nV
    eg = params.eg

    a = exp(bl-eg) * (exp(efp_s) - exp(efp_sm1))

    defp_s = (v_sm1 - v_s) / dx * exp(bl-eg+efp_s) * (exp(v_spN) - exp(v_s)) *\
             (exp(v_s) - exp(v_smN))

    defp_sm1 = - (v_sm1 - v_s) / dx * exp(bl-eg+efp_sm1) * (exp(v_spN) - exp(v_s)) *\
               (exp(v_s) - exp(v_smN))

    dv_s = a / dx * (- (exp(v_spN) - exp(v_s)) * (exp(v_s) - exp(v_smN)) \
                     - exp(v_s) * (exp(v_s) - exp(v_smN)) * (v_sm1 - v_s) \
                     + exp(v_s) * (exp(v_spN) - exp(v_s) * (v_sm1 - v_s)))
    
    dv_sm1 = a / dx

    dv_spN = a / dx * (v_sm1 - v_s) * exp(v_spN) * (exp(v_s) - exp(v_smN))

    dv_smN = - a / dx * (v_sm1 - v_s) * exp(v_smN) * (exp(v_spN) - exp(v_s))

    return nV*defp_sm1, nV*defp_s, nV*dv_smN, nV*dv_sm1, nV*dv_s, nV*dv_spN

def get_jvbpy_s_derivs(efp_s, efp_spN, v_sm1, v_s, v_smN, v_spN, dy, params):
    bl = params.bl
    nV = params.nV
    eg = params.eg

    a = exp(bl-eg) * (exp(efp_spN) - exp(efp_s))

    defp_s = - exp(bl-eg+efp_s) * (v_s-v_spN) * (exp(v_s) - exp(v_sm1)) *\
             (exp(v_s) - exp(v_smN)) / dy

    defp_spN = exp(bl-eg+efp_spN) * (v_s-v_spN) * (exp(v_s) - exp(v_sm1)) *\
             (exp(v_s) - exp(v_smN)) / dy

    dv_s = a / dy * ((exp(v_s) - exp(v_sm1)) * (exp(v_s) - exp(v_smN)) +\
    (v_s-v_spN) * exp(v_s) * ((exp(v_s) + exp(v_smN)) + (exp(v_s) - exp(v_sm1))))

    dv_sm1 = - a / dy * (v_s-v_spN) * exp(v_sm1) * (exp(v_s) - exp(v_smN))

    dv_spN = a / dy * (exp(v_s) - exp(v_sm1)) * (exp(v_s) - exp(v_smN))

    dv_smN = - a / dy * (v_s-v_spN) * exp(v_spN) * (exp(v_s) - exp(v_sm1))

    return nV*defp_s, nV*defp_spN, nV*dv_smN, nV*dv_sm1, nV*dv_s, nV*dv_spN
    
def get_jvbpy_smN_derivs(efp_smN, efp_s, v_sm1, v_s, v_smN, v_spN, dy, params):
    bl = params.bl
    nV = params.nV
    eg = params.eg

    a = exp(bl-eg) * (exp(efp_s) - exp(efp_smN))

    defp_s = exp(bl-eg+efp_s) * (v_smN-v_s) * (exp(v_s) - exp(v_sm1)) *\
             (exp(v_spN) - exp(v_s)) / dy

    defp_smN = - exp(bl-eg+efn_smN) * (v_smN-v_s) * (exp(v_s) - exp(v_sm1)) *\
               (exp(v_spN) - exp(v_s)) / dy

    dv_s = a / dy * (-(exp(v_s) - exp(v_sm1)) * (exp(v_spN) - exp(v_s)) +
    exp(v_s)*(v_smN-v_s) * ((exp(v_spN) - exp(v_s)) - (exp(v_s) - exp(v_sm1))))

    dv_sm1 = - a / dy * (v_smN-v_s) * (exp(v_spN) - exp(v_s)) * exp(v_sm1)

    dv_spN = a / dy * (v_smN-v_s) * (exp(v_s) - exp(v_sm1)) * exp(v_spN)

    dv_smN = a / dy * (exp(v_spN) - exp(v_s)) * (exp(v_s) - exp(v_sm1))

    return nV*defp_smN, nV*defp_s, nV*dv_smN, nV*dv_sm1, nV*dv_s, nV*dv_spN


def get_uvbp_derivs(n, p, efp_s, v_sm1, v_s, v_smN, v_spN, g, S, SGB, params):
    ni = params.ni
    n1 = params.n1
    p1 = params.p1
    nGB = params.nGB
    pGB = params.pGB

    r = S * (n*p - ni**2) / (n+p+n1+p1)
    rGB = SGB * (n*p - ni**2) / (n+p+nGB+pGB)

    defn = -(S * (n*p*(n1+p1+n+p) - (n*p-ni**2)*n) / (n1+p1+n+p)**2 +\
            SGB * (n*p*(nGB+pGB+n+p) - (n*p-ni**2)*n) / (nGB+pGB+n+p)**2) *\
            (exp(v_s)-exp(v_sm1)) * (exp(v_spN) - exp(v_s)) * (exp(v_s) - exp(v_smN))

    defp = -(S * (n*p*(n1+p1+n+p) - (n*p-ni**2)*p) / (n1+p1+n+p)**2 +\
            SGB * (n*p*(nGB+pGB+n+p) - (n*p-ni**2)*p) / (nGB+pGB+n+p)**2) *\
            (exp(v_s)-exp(v_sm1)) * (exp(v_spN) - exp(v_s)) * (exp(v_s) - exp(v_smN))

    dv_s = -(S * (n*p-ni**2) * (p-n) / (n1+p1+n+p)**2 +\
            SGB * (n*p-ni**2) * (p-n) / (nGB+pGB+n+p)**2) *\
           (exp(v_s)-exp(v_sm1)) * (exp(v_spN) - exp(v_s)) * (exp(v_s) - exp(v_smN))\
           - (g - r - rGB) * exp(v_s) * \
           ((exp(v_spN) - exp(v_s)) * (exp(v_s) - exp(v_smN))\
           + (exp(v_s)-exp(v_sm1)) * (exp(v_s) - exp(v_smN))\
           - (exp(v_s)-exp(v_sm1)) * (exp(v_spN) - exp(v_s)))
    
    dv_sm1 = -(g - r - rGB) * exp(v_sm1) * (exp(v_spN) - exp(v_s)) *\
              (exp(v_s) - exp(v_smN))

    dv_spN = (g - r - rGB) * exp(v_spN) * (exp(v_s) - exp(v_sm1)) *\
             (exp(v_s) - exp(v_smN))

    dv_smN = -(g - r - rGB) * exp(v_smN) * (exp(v_s) - exp(v_sm1)) *\
              (exp(v_spN) - exp(v_s))

    return defn, defp, dv_smN, dv_sm1, dv_s, dv_spN


def get_rr_derivs(efn_i, efp_i, v_i, n1, p1, S, params):
    ni = params.ni
    n = get_n(efn_i, v_i, params)
    p = get_p(efp_i, v_i, params)

    defp_i = S * (n*p*(n1+p1+n+p) - (n*p-ni**2)*p) / (n1+p1+n+p)**2
    defn_i = S * (n*p*(n1+p1+n+p) - (n*p-ni**2)*n) / (n1+p1+n+p)**2
    dv_i = S * (n*p-ni**2) * (p-n) / (n1+p1+n+p)**2

    return defp_i, defn_i, dv_i
