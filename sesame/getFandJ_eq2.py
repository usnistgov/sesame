# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import numpy as np
from itertools import chain, product

from .observables import get_n, get_p
from .defects  import defectsF, defectsJ
# remember that efn and efp are zero at equilibrium

def getFandJ_eq(sys, v, periodic_bcs, contacts_bcs):
    Nx, Ny = sys.xpts.shape[0], sys.ypts.shape[0]
    
    # lists of rows, columns and data that will create the sparse Jacobian
    rows = []
    columns = []
    data = []

    # right hand side vector
    vec = np.zeros((Nx*Ny,))

    ###########################################################################
    #                     organization of the Jacobian matrix                 #
    ###########################################################################
    # A site with coordinates (i,j) corresponds to a site number s as follows:
    # j = s//Nx
    # i = s - j*Nx
    #
    # Row for v_s
    # ----------------------------
    # fv_row = s
    #
    # Columns for v_s
    # -------------------------------
    # v_s_col = s
    # v_sp1_col = s+1
    # v_sm1_col = s-1
    # v_spN_col = s + Nx
    # v_smN_col = s - Nx

    def laplacian(vsmN, vsm1, vs, vsp1, vspN, dxm1, dx, dym1, dy, dxbar, dybar):
        res = ((vs - vsm1) / dxm1 - (vsp1 - vs) / dx) / dxbar\
            + ((vs - vsmN) / dym1 - (vspN - vs) / dy) / dybar
        return res

    ###########################################################################
    #                     For all sites in the system                         #
    ###########################################################################
    # carrier densities
    n = sys.Nc * np.exp(+sys.bl + v)
    p = sys.Nv * np.exp(-sys.Eg - sys.bl - v)

    # bulk charges
    rho = sys.rho - n + p
    drho_dv = -n - p

    # charge defects
    if len(sys.defects_list) != 0:
        defectsF(sys, n, p, rho)
        defectsJ(sys, n, p, drho_dv)

    # charge devided by epsilon
    #rho = rho / sys.epsilon
    #drho_dv = drho_dv / sys.epsilon

    # reshape the array as array[y-indices, x-indices]
    _sites = np.arange(Nx*Ny, dtype=int).reshape(Ny, Nx)

    ###########################################################################
    #       inside the system: 0 < i < Nx-1 and 0 < j < Ny-1                  #
    ###########################################################################
    # We compute fn, fp, fv derivatives. Those functions are only defined on the
    # inner part of the system. All the edges containing boundary conditions.

    # list of the sites inside the system
    sites = _sites[1:Ny-1, 1:Nx-1].flatten()

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny-2)
    dy = np.repeat(sys.dy[1:], Nx-2)
    dxm1 = np.tile(sys.dx[:-1], Ny-2)
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.

    #------------------------------ fv ----------------------------------------
    eps_m1x = .5 * (sys.epsilon[sites-1] + sys.epsilon[sites])
    eps_p1x = .5 * (sys.epsilon[sites+1] + sys.epsilon[sites])
    eps_m1y = .5 * (sys.epsilon[sites-Nx] + sys.epsilon[sites])
    eps_p1y = .5 * (sys.epsilon[sites+Nx] + sys.epsilon[sites])

    fv = (eps_m1x*(v[sites] - v[sites-1])/dxm1 - eps_p1x*(v[sites+1] - v[sites])/dx) / dxbar\
            + (eps_m1y*(v[sites] - v[sites-Nx])/dym1 - eps_p1y*(v[sites+Nx] - v[sites])/dy) / dybar\
            - rho[sites]

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #-------------------------- fv derivatives --------------------------------
    dvmN = -eps_m1y*1./(dym1 * dybar)
    dvm1 = -eps_m1x*1./(dxm1 * dxbar)
    dv = eps_m1x/(dxm1*dxbar) + eps_p1x/(dx*dxbar) + eps_m1y/(dym1*dybar) + eps_p1y/(dy*dybar) - drho_dv[sites]
    dvp1 = -eps_p1x*1./(dx * dxbar)
    dvpN = -eps_p1y*1./(dy * dybar)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = zip(sites, sites, sites, sites, sites)
    dfv_cols = zip(sites-Nx, sites-1, sites, sites+1, sites+Nx)
    dfv_data = zip(dvmN, dvm1, dv, dvp1, dvpN)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))


    ###########################################################################
    #                   left contact: i = 0 and 0 <= j <= Ny-1                #
    ###########################################################################
    # list of the sites on the left side
    sites = _sites[:, 0].flatten()

    if contacts_bcs == "Neumann":
        # update vector with no surface charges
        vec[sites] = v[sites+1]-v[sites]
        # update Jacobian
        dv = -np.ones(len(sites),)
        dvp1 = np.ones(len(sites),)
        dav_rows = zip(sites, sites)
        dav_cols = zip(sites, sites+1)
        dav_data = zip(dv, dvp1)

    if contacts_bcs == "Dirichlet":
        # update vector with zeros
        vec[sites] = 0
        # update Jacobian
        dav_rows = [sites.tolist()]
        dav_cols = [sites.tolist()]
        dav_data = [np.ones((len(sites,))).tolist()]

    rows += list(chain.from_iterable(dav_rows))
    columns += list(chain.from_iterable(dav_cols))
    data += list(chain.from_iterable(dav_data))


    ###########################################################################
    #                 right contact: i = Nx-1 and 0 <= j <= Ny-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[:, Nx-1].flatten()

    if contacts_bcs == "Neumann":
        # update vector with no surface charges
        vec[sites] = v[sites-1]-v[sites-2]
        # update Jacobian
        dv = np.ones(len(sites),)
        dvm1 = -np.ones(len(sites),)
        dbv_rows = zip(sites, sites)
        dbv_cols = zip(sites-1, sites)
        dbv_data = zip(dvm1, dv)

    if contacts_bcs == "Dirichlet":
        # update vector with zeros
        vec[sites] = 0
        # update Jacobian
        dbv_rows = [sites.tolist()]
        dbv_cols = [sites.tolist()]
        dbv_data = [np.ones((len(sites,))).tolist()]

    rows += list(chain.from_iterable(dbv_rows))
    columns += list(chain.from_iterable(dbv_cols))
    data += list(chain.from_iterable(dbv_data))

    ###########################################################################
    #                  boundary: 0 < i < Nx-1 and j = Ny-1                    #
    ###########################################################################
    if periodic_bcs:
        # We want periodic boundary conditions. This means that we can apply Poisson
        # equation assuming that the potential outside the system is the same as the
        # one on the opposite edge.

        # list of sites
        sites = _sites[Ny-1, 1:Nx-1].flatten()

        # lattice distances
        dx = sys.dx[1:]
        dxm1 = sys.dx[:-1]
        dy = np.repeat((sys.dy[0] + sys.dy[-1])/2, Nx-2)
        dym1 = np.repeat(sys.dy[-1], Nx-2)
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.

        eps_m1x = .5 * (sys.epsilon[sites - 1] + sys.epsilon[sites])
        eps_p1x = .5 * (sys.epsilon[sites + 1] + sys.epsilon[sites])
        eps_m1y = .5 * (sys.epsilon[sites - Nx] + sys.epsilon[sites])
        eps_p1y = .5 * (sys.epsilon[sites-Nx*(Ny-1)] + sys.epsilon[sites])

        #---------------------------------- fv -------------------------------------

        vsmN = v[sites-Nx]
        vsm1 = v[sites-1]
        vs = v[sites]
        vsp1 = v[sites+1]
        vspN = v[sites - Nx*(Ny-1)]

        fv = (eps_m1x * (v[sites] - v[sites - 1]) / dxm1 - eps_p1x * (v[sites + 1] - v[sites]) / dx) / dxbar \
             + (eps_m1y * (v[sites] - v[sites - Nx]) / dym1 - eps_p1y * (v[sites-Nx*(Ny-1)] - v[sites]) / dy) / dybar \
             - rho[sites]

        # update the vector rows for the inner part of the system
        vec[sites] = fv

        #-------------------------- fv derivatives --------------------------------
        dvmN = -eps_m1y * 1. / (dym1 * dybar)
        dvm1 = -eps_m1x * 1. / (dxm1 * dxbar)
        dv = eps_m1x / (dxm1 * dxbar) + eps_p1x / (dx * dxbar) + eps_m1y / (dym1 * dybar) + eps_p1y / (dy * dybar) - drho_dv[sites]
        dvp1 = -eps_p1x * 1. / (dx * dxbar)
        dvpN = -eps_p1y * 1. / (dy * dybar)

        # update the sparse matrix row and columns
        dfv_rows = zip(sites, sites, sites, sites, sites)
        dfv_cols = zip(sites-Nx*(Ny-1), sites-Nx, sites-1, sites, sites+1)
        dfv_data = zip(dvpN, dvmN, dvm1, dv, dvp1)

    else:

        # list of sites
        sites = _sites[Ny - 1, 1:Nx - 1].flatten()

        # lattice distances
        dx = sys.dx[1:]
        dxm1 = sys.dx[:-1]
        dy = 0
        dym1 = np.repeat(sys.dy[-1], Nx-2)
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.

        eps_m1x = .5 * (sys.epsilon[sites - 1] + sys.epsilon[sites])
        eps_p1x = .5 * (sys.epsilon[sites + 1] + sys.epsilon[sites])
        eps_m1y = .5 * (sys.epsilon[sites - Nx] + sys.epsilon[sites])

        #---------------------------------- fv -------------------------------------
        fv = (eps_m1x * (v[sites] - v[sites - 1]) / dxm1 - eps_p1x * (v[sites + 1] - v[sites]) / dx) / dxbar \
             + (eps_m1y * (v[sites] - v[sites - Nx]) / dym1) / dybar \
             - rho[sites]

        # update the vector rows for the inner part of the system
        vec[sites] = fv

        #-------------------------- fv derivatives --------------------------------
        dvmN = -eps_m1y * 1. / (dym1 * dybar)
        dvm1 = -eps_m1x * 1. / (dxm1 * dxbar)
        dv = eps_m1x / (dxm1 * dxbar) + eps_p1x / (dx * dxbar) + eps_m1y / (dym1 * dybar)  - \
             drho_dv[sites]
        dvp1 = -eps_p1x * 1. / (dx * dxbar)

        # update the sparse matrix row and columns
        dfv_rows = zip(sites, sites, sites, sites)
        dfv_cols = zip(sites-Nx, sites-1, sites, sites+1)
        dfv_data = zip(dvmN, dvm1, dv, dvp1)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))

    ###########################################################################
    #                     boundary: 0 < i < Nx-1 and j = 0                    #
    ###########################################################################
    if periodic_bcs:
        # list of sites
        sites = _sites[0, 1:Nx-1].flatten()

        # dxbar and dybar
        dx = sys.dx[1:]
        dxm1 = sys.dx[:-1]
        dy = np.repeat(sys.dy[0], Nx-2)
        dym1 = np.repeat((sys.dy[0] + sys.dy[-1])/2, Nx-2)
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.

        eps_m1x = .5 * (sys.epsilon[sites - 1] + sys.epsilon[sites])
        eps_p1x = .5 * (sys.epsilon[sites + 1] + sys.epsilon[sites])
        eps_m1y = .5 * (sys.epsilon[sites+Nx*(Ny-1)] + sys.epsilon[sites])
        eps_p1y = .5 * (sys.epsilon[sites + Nx] + sys.epsilon[sites])

        #---------------------------------- fv -------------------------------------
        vsmN = v[sites + Nx*(Ny-1)]
        vsm1 = v[sites-1]
        vs = v[sites]
        vsp1 = v[sites+1]
        vspN = v[sites+Nx]

        fv = (eps_m1x * (v[sites] - v[sites - 1]) / dxm1 - eps_p1x * (v[sites + 1] - v[sites]) / dx) / dxbar \
             + (eps_m1y * (v[sites] - v[sites + Nx * (Ny - 1)]) / dym1 - eps_p1y * (v[sites + Nx] - v[sites]) / dy) / dybar \
             - rho[sites]

        # update the vector rows for the inner part of the system
        vec[sites] = fv

        #-------------------------- fv derivatives --------------------------------
        dvmN = -eps_m1y * 1. / (dym1 * dybar)
        dvm1 = -eps_m1x * 1. / (dxm1 * dxbar)
        dv = eps_m1x / (dxm1 * dxbar) + eps_p1x / (dx * dxbar) + eps_m1y / (dym1 * dybar) + eps_p1y / (dy * dybar) - \
             drho_dv[sites]
        dvp1 = -eps_p1x * 1. / (dx * dxbar)
        dvpN = -eps_p1y * 1. / (dy * dybar)

        # update the sparse matrix row and columns
        dfv_rows = zip(sites, sites, sites, sites, sites)
        dfv_cols = zip(sites-1, sites, sites+1, sites+Nx, sites+Nx*(Ny-1))
        dfv_data = zip(dvm1, dv, dvp1, dvpN, dvmN)

    else:
        # list of sites
        sites = _sites[0, 1:Nx - 1].flatten()

        # dxbar and dybar
        dx = sys.dx[1:]
        dxm1 = sys.dx[:-1]
        dy = np.repeat(sys.dy[0], Nx-2)
        dym1 = 0
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.

        eps_m1x = .5 * (sys.epsilon[sites - 1] + sys.epsilon[sites])
        eps_p1x = .5 * (sys.epsilon[sites + 1] + sys.epsilon[sites])
        eps_p1y = .5 * (sys.epsilon[sites + Nx] + sys.epsilon[sites])

        #---------------------------------- fv -------------------------------------
        fv = (eps_m1x * (v[sites] - v[sites - 1]) / dxm1 - eps_p1x * (v[sites + 1] - v[sites]) / dx) / dxbar \
             + (- eps_p1y * (v[sites + Nx] - v[sites]) / dy) / dybar \
             - rho[sites]

        # update the vector rows for the inner part of the system
        vec[sites] = fv

        #-------------------------- fv derivatives --------------------------------
        dvm1 = -eps_m1x * 1. / (dxm1 * dxbar)
        dv = eps_m1x / (dxm1 * dxbar) + eps_p1x / (dx * dxbar) + eps_p1y / (dy * dybar) - \
             drho_dv[sites]
        dvp1 = -eps_p1x * 1. / (dx * dxbar)
        dvpN = -eps_p1y * 1. / (dy * dybar)

        # update the sparse matrix row and columns
        dfv_rows = zip(sites, sites, sites, sites)
        dfv_cols = zip(sites-1, sites, sites+1, sites+Nx)
        dfv_data = zip(dvm1, dv, dvp1, dvpN)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))

    return vec, rows, columns, data
