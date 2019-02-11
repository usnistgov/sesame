# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import numpy as np
from .observables import *
from .defects import defectsF


def getF(sys, v, efn, efp, veq):
    ###########################################################################
    #               organization of the right hand side vector                #
    ###########################################################################
    # A site with coordinates (i,j) corresponds to a site number s as follows:
    # j = s//Nx
    # i = s - j*Nx --> (i = s % Nx)
    #
    # Rows for (efn_s, efp_s, v_s)
    # ----------------------------
    # fn_row = 3*s
    # fp_row = 3*s+1
    # fv_row = 3*s+2

    Nx, Ny = sys.xpts.shape[0], sys.ypts.shape[0]
    N = Nx* Ny
    dx, dy = sys.dx, sys.dy

    # right hand side vector
    vec = np.zeros((3 * Nx * Ny,))

    ###########################################################################
    #                     For all sites in the system                         #
    ###########################################################################
    # carrier densities
    n = sys.Nc * np.exp(+sys.bl + efn + v)
    p = sys.Nv * np.exp(-sys.Eg - sys.bl - efp - v)

    # equilibrium carrier densities
    n_eq = sys.Nc * np.exp(+sys.bl + veq)
    p_eq = sys.Nv * np.exp(-sys.Eg - sys.bl - veq)

    # bulk charges
    rho = sys.rho - n + p

    # recombination rates
    r = get_bulk_rr(sys, n, p)

    # charge defects
    if len(sys.defects_list) != 0:
        defectsF(sys, sys.defects_list, n, p, rho, r)

    # reshape the array as array[y-indices, x-indices]
    _sites = np.arange(Nx * Ny, dtype=int).reshape(Ny, Nx)

    ###########################################################################
    #       inside the system: 0 < i < Nx-1 and 0 <= j <= Ny-1                #
    ###########################################################################
    # We compute fn, fp, fv. Those functions are only defined on the
    # inner part of the system. All the edges containing boundary conditions.

    # list of the sites inside the system
    sites = _sites[0:Ny, 1:Nx - 1].flatten()

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny)
    dxm1 = np.tile(sys.dx[:-1], Ny)

    dy = np.repeat(sys.dy,Nx-2)
    dym1 = np.repeat(np.roll(sys.dy,1),Nx-2)

    dxbar = (dxm1 + dx) / 2.
    dybar = (dym1 + dy) / 2.

    infind = np.where(np.isinf(dybar))
    for i in infind[0]:
        if np.isinf(dy[i]):
            dybar[i] = dy[i-Nx] / 2.
        else:
            dybar[i] = dy[i] / 2.


    # compute the currents
    jnx_s = get_jn(sys, efn, v, sites, sites + 1, dx)
    jnx_sm1 = get_jn(sys, efn, v, sites - 1, sites, dxm1)
    jny_s = get_jn(sys, efn, v, sites, (sites + Nx) % N, dy)
    jny_smN = get_jn(sys, efn, v, (sites - Nx) % N, sites, dym1)

    jpx_s = get_jp(sys, efp, v, sites, sites + 1, dx)
    jpx_sm1 = get_jp(sys, efp, v, sites - 1, sites, dxm1)
    jpy_s = get_jp(sys, efp, v, sites, (sites + Nx) % N, dy)
    jpy_smN = get_jp(sys, efp, v, (sites - Nx) % N, sites, dym1)

    # ------------------------------ fn ----------------------------------------
    fn = (jnx_s - jnx_sm1) / dxbar + (jny_s - jny_smN) / dybar \
         + sys.g[sites] - r[sites]

    vec[3 * sites] = fn

    # ------------------------------ fp ----------------------------------------
    fp = (jpx_s - jpx_sm1) / dxbar + (jpy_s - jpy_smN) / dybar \
         + r[sites] - sys.g[sites]

    vec[3 * sites + 1] = fp

    # ------------------------------ fv ----------------------------------------
    eps_m1x = .5 * (sys.epsilon[sites - 1] + sys.epsilon[sites])
    eps_p1x = .5 * (sys.epsilon[sites + 1] + sys.epsilon[sites])
    eps_m1y = .5 * (sys.epsilon[(sites - Nx)%N] + sys.epsilon[sites])
    eps_p1y = .5 * (sys.epsilon[(sites + Nx)%N] + sys.epsilon[sites])

    fv = (eps_m1x * (v[sites] - v[sites - 1]) / dxm1 - eps_p1x * (v[sites + 1] - v[sites]) / dx) / dxbar \
         + (eps_m1y * (v[sites] - v[(sites - Nx)%N]) / dym1 - eps_p1y * (v[(sites + Nx)%N] - v[sites]) / dy) / dybar \
         - rho[sites]

    vec[3 * sites + 2] = fv

    ###########################################################################
    #                 left boundary: i = 0 and 0 <= j <= Ny-1                 #
    ###########################################################################
    # list of the sites on the left side
    sites = _sites[:, 0].flatten()

    # compute the currents
    # s_sp1 = [i for i in zip(sites, sites + 1)]
    jnx = get_jn(sys, efn, v, sites, sites + 1, sys.dx[0])
    jpx = get_jp(sys, efp, v, sites, sites + 1, sys.dx[0])

    # compute an, ap, av
    an = jnx - sys.Scn[0] * (n[sites] - n_eq[sites])
    ap = jpx + sys.Scp[0] * (p[sites] - p_eq[sites])
    av = 0  # to ensure Dirichlet BCs

    vec[3 * sites] = an
    vec[3 * sites + 1] = ap
    vec[3 * sites + 2] = av

    ###########################################################################
    #               right boundary: i = Nx-1 and 0 <= j <= Ny-1                 #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[:, Nx - 1].flatten()

    # currents
    jnx_sm1 = get_jn(sys, efn, v, sites - 1, sites, sys.dx[-1])
    jpx_sm1 = get_jp(sys, efp, v, sites - 1, sites, sys.dx[-1])

    # b_n, b_p and b_v values
    bn = jnx_sm1 + sys.Scn[1] * (n[sites] - n_eq[sites])
    bp = jpx_sm1 - sys.Scp[1] * (p[sites] - p_eq[sites])
    bv = 0  # Dirichlet BC

    vec[3 * sites] = bn
    vec[3 * sites + 1] = bp
    vec[3 * sites + 2] = bv

    return vec
