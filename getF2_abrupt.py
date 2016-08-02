import numpy as np

from sesame.observables import *

def getF(sys, v, efn, efp):
    ###########################################################################
    #               organization of the right hand side vector                #
    ###########################################################################
    # A site with coordinates (i,j) corresponds to a site number s as follows:
    # j = s//Nx
    # i = s - j*Nx
    #
    # Rows for (efn_s, efp_s, v_s)
    # ----------------------------
    # fn_row = 3*s
    # fp_row = 3*s+1
    # fv_row = 3*s+2

    Nx, Ny = sys.xpts.shape[0], sys.ypts.shape[0]
    dx, dy = sys.dx, sys.dy

    # right hand side vector
    vec = np.zeros((3*Nx*Ny,))

    ###########################################################################
    #                     For all sites in the system                         #
    ###########################################################################
    sites = [i + j*Nx for j in range(Ny) for i in range(Nx)]

    # carrier densities
    n = get_n(sys, efn, v, sites)
    p = get_p(sys, efp, v, sites)

    # bulk charges
    rho = sys.rho - n + p

    # recombination rates
    r = get_rr(sys, n, p, sys.n1, sys.p1, sys.tau_e, sys.tau_h, sites)

    # extra charge density
    if hasattr(sys, 'Nextra'): 
        # find sites containing extra charges
        matches = sys.extra_charge_sites

        nextra = sys.nextra[matches]
        pextra = sys.pextra[matches]
        _n = n[matches]
        _p = p[matches]

        # extra charge density
        f = (_n + pextra) / (_n + _p + nextra + pextra)
        rho[matches] += sys.Nextra[matches] / 2. * (1 - 2*f)

        # extra charge recombination
        r[matches] += get_rr(sys, _n, _p, nextra, pextra, 1/sys.Seextra[matches],
                             1/sys.Shextra[matches], matches)

    
    # charge devided by epsilon
    rho = rho / sys.epsilon[sites]

    ###########################################################################
    #       inside the system: 0 < i < Nx-1 and 0 < j < Ny-1                  #
    ###########################################################################
    # We compute fn, fp, fv. Those functions are only defined on the
    # inner part of the system.

    # list of the sites inside the system
    sites = [i + j*Nx for j in range(1,Ny-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny-2)
    dxm1 = np.tile(sys.dx[:-1], Ny-2)
    dy = np.repeat(sys.dy[1:], Nx-2)
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dxbar = (dxm1 + dx) / 2.
    dybar = (dym1 + dy) / 2.

    # compute the currents
    jnx_s   = get_jn(sys, efn, v, sites, sites+1, dx)
    jnx_sm1 = get_jn(sys, efn, v, sites-1, sites, dxm1)
    jny_s   = get_jn(sys, efn, v, sites, sites+Nx, dy)
    jny_smN = get_jn(sys, efn, v, sites-Nx, sites, dym1)

    jpx_s   = get_jp(sys, efp, v, sites, sites+1, dx)
    jpx_sm1 = get_jp(sys, efp, v, sites-1, sites, dxm1)
    jpy_s   = get_jp(sys, efp, v, sites, sites+Nx, dy)
    jpy_smN = get_jp(sys, efp, v, sites-Nx, sites, dym1)


    #------------------------------ fn ----------------------------------------
    fn = (jnx_s - jnx_sm1) / dxbar + (jny_s - jny_smN) / dybar \
       + sys.g[sites] - r[sites]

    vec[3*sites] = fn

    #------------------------------ fp ----------------------------------------
    fp = (jpx_s - jpx_sm1) / dxbar + (jpy_s - jpy_smN) / dybar \
       + r[sites] - sys.g[sites]

    vec[3*sites+1] = fp

    #------------------------------ fv ----------------------------------------
    fv = ((v[sites]-v[sites-1]) / dxm1 - (v[sites+1]-v[sites]) / dx) / dxbar\
       + ((v[sites]-v[sites-Nx]) / dym1 - (v[sites+Nx]-v[sites]) / dy) / dybar\
       - rho[sites]

    vec[3*sites+2] = fv

    ###########################################################################
    #                 left boundary: i = 0 and 0 <= j <= Ny-1                 #
    ###########################################################################
    # list of the sites on the left side
    sites = [j*Nx for j in range(Ny)]
    sites = np.asarray(sites)

    # compute the currents
    jnx = get_jn(sys, efn, v, sites, sites+1, sys.dx[0])
    jpx = get_jp(sys, efp, v, sites, sites+1, sys.dx[0])

    # compute an, ap, av
    n_eq = 0
    p_eq = 0
    #TODO tricky here to decide
    if sys.rho[Nx] < 0: # p doped
        p_eq = -sys.rho[sites]
        n_eq = sys.ni[sites]**2 / p_eq
    else: # n doped
        n_eq = sys.rho[sites]
        p_eq = sys.ni[sites]**2 / n_eq

    an = jnx - sys.Scn[0] * (n[sites] - n_eq)
    ap = jpx + sys.Scp[0] * (p[sites] - p_eq)
    av = 0 # to ensure Dirichlet BCs
    #
    vec[3*sites] = an
    vec[3*sites+1] = ap
    vec[3*sites+2] = av

    
    ###########################################################################
    #               right boundary: i = Nx-1 and 0 < j < Ny-1                 #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + j*Nx for j in range(1,Ny-1)]
    sites = np.asarray(sites)

    # dxbar and dybar
    dxm1 = sys.dx[-1]
    dy = sys.dy[1:]
    dym1 = sys.dy[:-1]
    dxbar = np.tile(sys.dx[-1], Ny-2)
    dybar = (dy + dym1) / 2.

    # currents
    jnx_sm1 = get_jn(sys, efn, v, sites-1, sites, dxm1)
    jny_s   = get_jn(sys, efn, v, sites, sites+Nx, dy)
    jny_smN = get_jn(sys, efn, v, sites-Nx, sites, dym1)

    jpx_sm1 = get_jp(sys, efp, v, sites-1, sites, dxm1)
    jpy_s   = get_jp(sys, efp, v, sites, sites+Nx, dy)
    jpy_smN = get_jp(sys, efp, v, sites-Nx, sites, dym1)

    jnx_s = jnx_sm1 + dxbar * (r[sites] - sys.g[sites] - (jny_s - jny_smN)/dybar)
    jpx_s = jpx_sm1 + dxbar * (sys.g[sites] - r[sites] - (jpy_s - jpy_smN)/dybar)

    # b_n, b_p and b_v values
    n_eq = 0
    p_eq = 0
    if sys.rho[2*Nx-1] < 0: # p doped
        p_eq = -sys.rho[2*Nx-1]
        n_eq = sys.ni[sites]**2 / p_eq
    else: # n doped
        n_eq = sys.rho[2*Nx-1]
        p_eq = sys.ni[sites]**2 / n_eq

    bn = jnx_s + sys.Scn[1] * (n[sites] - n_eq)
    bp = jpx_s - sys.Scp[1] * (p[sites] - p_eq)
    bv = 0 # Dirichlet BC

    vec[3*sites] = bn
    vec[3*sites+1] = bp
    vec[3*sites+2] = bv      

    ###########################################################################
    #                    right boundary: i = Nx-1 and j = 0                   #
    ###########################################################################
    # list of the sites
    sites = np.array([Nx-1])

    # dxbar and dybar
    dxm1 = sys.dx[-1]
    dy = sys.dy[0]
    dym1 = 0
    dxbar = sys.dx[-1]
    dybar = (dy + dym1) / 2.

    # compute the currents
    jnx_sm1 = get_jn(sys, efn, v, Nx-2, Nx-1, dxm1)
    jny_s   = get_jn(sys, efn, v, Nx-1, 2*Nx-1, dy)
    jny_smN = 0

    jpx_sm1 = get_jp(sys, efp, v, Nx-2, Nx-1, dxm1)
    jpy_s   = get_jp(sys, efp, v, Nx-1, 2*Nx-1, dy)
    jpy_smN = 0

    jnx_s = jnx_sm1 + dxbar * (r[sites] - sys.g[sites] - (jny_s - jny_smN)/dybar)
    jpx_s = jpx_sm1 + dxbar * (sys.g[sites] - r[sites] - (jpy_s - jpy_smN)/dybar)

    # b_n, b_p and b_v values
    n_eq = 0
    p_eq = 0
    if sys.rho[2*Nx-1] < 0: # p doped
        p_eq = -sys.rho[2*Nx-1]
        n_eq = sys.ni[sites]**2 / p_eq
    else: # n doped
        n_eq = sys.rho[2*Nx-1]
        p_eq = sys.ni[sites]**2 / n_eq

    bn = jnx_s + sys.Scn[1] * (n[sites] - n_eq)
    bp = jpx_s - sys.Scp[1] * (p[sites] - p_eq)
    bv = 0 # Dirichlet BC

    vec[3*sites] = bn
    vec[3*sites+1] = bp
    vec[3*sites+2] = bv 

    ###########################################################################
    #                 right boundary: i = Nx-1 and j = Ny-1                   #
    ###########################################################################
    # list of the sites
    sites = np.array([Nx*Ny-1])

    # dxbar and dybar
    dxm1 = sys.dx[-1]
    dy = 0
    dym1 = sys.dy[-1]
    dxbar = sys.dx[-1]
    dybar = (dy + dym1) / 2.

    # compute the currents
    jnx_sm1 = get_jn(sys, efn, v, Nx*Ny-2, Nx*Ny-1, dxm1)
    jny_s   = 0
    jny_smN = get_jn(sys, efn, v, Nx*(Ny-1)-1, Nx*Ny-1, dym1)

    jpx_sm1 = get_jp(sys, efp, v, Nx*Ny-2, Nx*Ny-1, dxm1)
    jpy_s   = 0
    jpy_smN = get_jp(sys, efp, v, Nx*(Ny-1)-1, Nx*Ny-1, dym1)

    jnx_s = jnx_sm1 + dxbar * (r[sites] - sys.g[sites] - (jny_s - jny_smN)/dybar)
    jpx_s = jpx_sm1 + dxbar * (sys.g[sites] - r[sites] - (jpy_s - jpy_smN)/dybar)

    # b_n, b_p and b_v values
    n_eq = 0
    p_eq = 0
    if sys.rho[2*Nx-1] < 0: # p doped
        p_eq = -sys.rho[2*Nx-1]
        n_eq = sys.ni[sites]**2 / p_eq
    else: # n doped
        n_eq = sys.rho[2*Nx-1]
        p_eq = sys.ni[sites]**2 / n_eq

    bn = jnx_s + sys.Scn[1] * (n[sites] - n_eq)
    bp = jpx_s - sys.Scp[1] * (p[sites] - p_eq)
    bv = 0 # Dirichlet BC

    vec[3*sites] = bn
    vec[3*sites+1] = bp
    vec[3*sites+2] = bv 

    ###########################################################################
    #               bottom boundary: 0 < i < Nx-1 and j = 0                   #
    ###########################################################################
    # We compute fn, fp, fv. We apply drift diffusion equations

    # list of the sites inside the system
    sites = [i for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = sys.dx[1:]
    dxm1 = sys.dx[:-1]
    dy = np.repeat(sys.dy[0], Nx-2)
    dym1 = 0
    dxbar = (dxm1 + dx) / 2.
    dybar = (dym1 + dy) / 2.

    # compute the currents
    jnx_s   = get_jn(sys, efn, v, sites, sites + 1, dx)
    jnx_sm1 = get_jn(sys, efn, v, sites - 1, sites, dxm1)
    jny_s   = get_jn(sys, efn, v, sites, sites + Nx, dy)
    jny_smN = 0

    jpx_s   = get_jp(sys, efp, v, sites, sites + 1, dx)
    jpx_sm1 = get_jp(sys, efp, v, sites - 1, sites, dxm1)
    jpy_s   = get_jp(sys, efp, v, sites, sites + Nx, dy)
    jpy_smN = 0

    #------------------------------ fn ----------------------------------------
    fn = (jnx_s - jnx_sm1) / dxbar + (jny_s - jny_smN) / dybar \
            + sys.g[sites] - r[sites]

    vec[3*sites] = fn

    #------------------------------ fp ----------------------------------------
    fp = (jpx_s - jpx_sm1) / dxbar + (jpy_s - jpy_smN) / dybar \
        + r[sites] - sys.g[sites]

    vec[3*sites+1] = fp

    #------------------------------ fv ----------------------------------------
    fv = ((v[sites]-v[sites-1]) / dxm1 - (v[sites+1]-v[sites]) / dx) / dxbar\
       + (-(v[sites+Nx]-v[sites]) / dy) / dybar\
       - rho[sites]

    vec[3*sites+2] = fv

    ###########################################################################
    #                top  boundary: 0 < i < Nx-1 and j = Ny-1                 #
    ###########################################################################
    # We compute fn, fp, fv. We apply drift diffusion equations

    # list of the sites inside the system
    sites = [i + (Ny-1)*Nx for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = sys.dx[1:]
    dxm1 = sys.dx[:-1]
    dy = 0
    dym1 = np.repeat(sys.dy[-1], Nx-2)
    dxbar = (dxm1 + dx) / 2.
    dybar = (dym1 + dy) / 2.

    # compute the currents
    jnx_s   = get_jn(sys, efn, v, sites, sites + 1, dx)
    jnx_sm1 = get_jn(sys, efn, v, sites - 1, sites, dxm1)
    jny_s   = 0
    jny_smN = get_jn(sys, efn, v, sites - Nx, sites, dym1)

    jpx_s   = get_jp(sys, efp, v, sites, sites + 1, dx)
    jpx_sm1 = get_jp(sys, efp, v, sites - 1, sites, dxm1)
    jpy_s   = 0
    jpy_smN = get_jp(sys, efp, v, sites - Nx, sites, dym1)

    #------------------------------ fn ----------------------------------------
    fn = (jnx_s - jnx_sm1) / dxbar + (jny_s - jny_smN) / dybar \
       + sys.g[sites] - r[sites]

    vec[3*sites] = fn

    #------------------------------ fp ----------------------------------------
    fp = (jpx_s - jpx_sm1) / dxbar + (jpy_s - jpy_smN) / dybar \
       + r[sites] - sys.g[sites]

    vec[3*sites+1] = fp

    #------------------------------ fv ----------------------------------------
    fv = ((v[sites]-v[sites-1]) / dxm1 - (v[sites+1]-v[sites]) / dx) / dxbar\
       + ((v[sites]-v[sites-Nx]) / dym1) / dybar\
       - rho[sites]

    vec[3*sites+2] = fv

    return vec
