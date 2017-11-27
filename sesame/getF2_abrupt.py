# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import numpy as np
from .observables import *
from .defects  import defectsF

def getF(sys, v, efn, efp, veq):
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
    # carrier densities
    n = sys.Nc * np.exp(+sys.bl + efn + v)
    p = sys.Nv * np.exp(-sys.Eg - sys.bl + efp - v)

    # equilibrium carrier densities
    n_eq = sys.Nc * np.exp(+sys.bl + veq)
    p_eq = sys.Nv * np.exp(-sys.Eg - sys.bl - veq)

    # bulk charges
    rho = sys.rho - n + p

    # recombination rates
    r = get_bulk_rr(sys, n, p)

    # charge defects
    if len(sys.defects_list) != 0:
        defectsF(sys, n, p, rho, r)

    # charge devided by epsilon
    rho = rho / sys.epsilon

    # reshape the array as array[y-indices, x-indices]
    _sites = np.arange(Nx*Ny, dtype=int).reshape(Ny, Nx)

    ###########################################################################
    #       inside the system: 0 < i < Nx-1 and 0 < j < Ny-1                  #
    ###########################################################################
    # We compute fn, fp, fv. Those functions are only defined on the
    # inner part of the system.

    # list of the sites inside the system
    sites = _sites[1:Ny-1, 1:Nx-1].flatten()

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
    sites = _sites[:, 0].flatten()

    # compute the currents
    jnx = get_jn(sys, efn, v, sites, sites+1, sys.dx[0])
    jpx = get_jp(sys, efp, v, sites, sites+1, sys.dx[0])

    # compute an, ap, av
    an = jnx - sys.Scn[0] * (n[sites] - n_eq[sites])
    ap = jpx + sys.Scp[0] * (p[sites] - p_eq[sites])
    av = 0 # to ensure Dirichlet BCs
    #
    vec[3*sites] = an
    vec[3*sites+1] = ap
    vec[3*sites+2] = av

    
    ###########################################################################
    #               right boundary: i = Nx-1 and 0 < j < Ny-1                 #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[1:Ny-1, Nx-1].flatten()

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
    bn = jnx_s + sys.Scn[1] * (n[sites] - n_eq[sites])
    bp = jpx_s - sys.Scp[1] * (p[sites] - p_eq[sites])
    bv = 0 # Dirichlet BC

    vec[3*sites] = bn
    vec[3*sites+1] = bp
    vec[3*sites+2] = bv      

    ###########################################################################
    #                    right boundary: i = Nx-1 and j = 0                   #
    ###########################################################################
    # list of the sites
    sites = _sites[0, Nx-1].flatten()

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
    bn = jnx_s + sys.Scn[1] * (n[sites] - n_eq[sites])
    bp = jpx_s - sys.Scp[1] * (p[sites] - p_eq[sites])
    bv = 0 # Dirichlet BC

    vec[3*sites] = bn
    vec[3*sites+1] = bp
    vec[3*sites+2] = bv 

    ###########################################################################
    #                 right boundary: i = Nx-1 and j = Ny-1                   #
    ###########################################################################
    # list of the sites
    sites = _sites[Ny-1, Nx-1].flatten()

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
    bn = jnx_s + sys.Scn[1] * (n[sites] - n_eq[sites])
    bp = jpx_s - sys.Scp[1] * (p[sites] - p_eq[sites])
    bv = 0 # Dirichlet BC

    vec[3*sites] = bn
    vec[3*sites+1] = bp
    vec[3*sites+2] = bv 

    ###########################################################################
    #               bottom boundary: 0 < i < Nx-1 and j = 0                   #
    ###########################################################################
    # We compute fn, fp, fv. We apply drift diffusion equations

    # list of the sites inside the system
    sites = _sites[0, 1:Nx-1]

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
    sites = _sites[Ny-1, 1:Nx-1]

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
