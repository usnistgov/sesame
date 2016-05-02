import numpy as np
from numpy import exp
from scipy.sparse import coo_matrix
from itertools import chain

from sesame.observables2 import *

def getJ(sys, v, efn, efp):
    ###########################################################################
    #                     organization of the Jacobian matrix                 #
    ###########################################################################
    # A site with coordinates (i,j,k) corresponds to a site number s as follows:
    # k = s//(Nx*Ny)
    # j = s - s//Nx
    # i = s - j*Nx - k*Nx*Ny

    #
    # Rows for (efn_s, efp_s, v_s)
    # ----------------------------
    # fn_row = 3*s
    # fp_row = 3*s+1
    # fv_row = 3*s+2
    #
    # Columns for (efn_s, efp_s, v_s)
    # -------------------------------
    # efn_smN_col = 3*(s-Nx)
    # efn_sm1_col = 3*(s-1)
    # efn_s_col = 3*s
    # efn_sp1_col = 3*(s+1)
    # efn_spN_col = 3*(s+Nx)
    #
    # efp_smN_col = 3*(s-Nx)+1
    # efp_sm1_col = 3*(s-1)+1
    # efp_s_col = 3*s+1
    # efp_sp1_col = 3*(s+1)+1
    # efp_spN_col = 3*(s+Nx)+1
    #
    # v_smN_col = 3*(s-Nx)+2
    # v_sm1_col = 3*(s-1)+2
    # v_s_col = 3*s+2
    # v_sp1_col = 3*(s+1)+2
    # v_spN_col = 3*(s+Nx)+2

    Nx, Ny, Nz = sys.xpts.shape[0], sys.ypts.shape[0], sys.zpts.shape[0]

    # lists of rows, columns and data that will create the sparse Jacobian
    global rows, columns, data
    rows = []
    columns = []
    data = []

    ###########################################################################
    #                     For all sites in the system                         #
    ###########################################################################
    sites = [i + j*Nx + k*Nx*Ny for k in range(Nz) for j in range(Ny) for i in range(Nx)]

    # carrier densities
    n = get_n(sys, efn, v, sites)
    p = get_p(sys, efp, v, sites)

    # bulk charges
    drho_defn_s = - n
    drho_defp_s = p
    drho_dv_s = - n - p

    # derivatives of the bulk recombination rates
    dr_defn_s, dr_defp_s, dr_dv_s = \
    get_rr_derivs(sys, n, p, sys.n1, sys.p1, sys.tau_e, sys.tau_h, sites)\

    # extra charge density
    if hasattr(sys, 'Nextra'): 
        # find sites containing extra charges
        matches = [s for s in sites if s in sys.extra_charge_sites]

        nextra = sys.nextra[matches]
        pextra = sys.pextra[matches]
        _n = n[matches]
        _p = p[matches]

        # extra charge density
        drho_defn_s[matches] += - sys.Nextra[matches] \
                                * (_n*(_n+_p+nextra+pextra)-(_n+pextra)*_n)\
                                / (_n+_p+nextra+pextra)**2
        drho_defp_s[matches] += sys.Nextra[matches] * (_n+pextra)*_p \
                              / (_n+_p+nextra+pextra)**2
        drho_dv[matches] += - sys.Nextra[matches]\
                            * (_n*(_n+_p+nextra+pextra)-(_n+pextra)*(_n-_p))\
                            / (_n+_p+nextra+pextra)**2

        # extra charge recombination
        defn, defp, dv =  get_rr_derivs(sys, _n, _p, nextra, pextra, 1/sys.Sextra[matches], 
                                        1/sys.Sextra[matches], matches)
        dr_defn_s[matches] += defn
        dr_defp_s[matches] += defp
        dr_dv_s[matches] += dv

    # charge is divided by epsilon
    drho_defn_s = drho_defn_s / sys.epsilon[sites]
    drho_defp_s = drho_defp_s / sys.epsilon[sites]
    drho_dv_s = drho_dv_s / sys.epsilon[sites]

    def update(r, c, d):
        global rows, columns, data
        rows += list(chain.from_iterable(r))
        columns += list(chain.from_iterable(c))
        data += list(chain.from_iterable(d))


    def fn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN,\
                       dx, dxm1, dy, dym1, dz, dzm1, sites):

        # lattice distances
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        # pairs of sites in the x-direction (always defined like this)
        sm1_s = [i for i in zip(sites - 1, sites)]
        s_sp1 = [i for i in zip(sites, sites + 1)]

        # get the derivatives of all currents
        djx_s_defn_s, djx_s_defn_sp1, djx_s_dv_s, djx_s_dv_sp1 = \
        get_jn_derivs(sys, efn, v, s_sp1, dx)

        djx_sm1_defn_sm1, djx_sm1_defn_s, djx_sm1_dv_sm1, djx_sm1_dv_s = \
        get_jn_derivs(sys, efn, v, sm1_s, dxm1)

        djy_s_defn_s, djy_s_defn_spN, djy_s_dv_s, djy_s_dv_spN = \
        get_jn_derivs(sys, efn, v, s_spN, dy)

        djy_smN_defn_smN, djy_smN_defn_s, djy_smN_dv_smN, djy_smN_dv_s = \
        get_jn_derivs(sys, efn, v, smN_s, dym1)

        djz_s_defn_s, djz_s_defn_spNN, djz_s_dv_s, djz_s_dv_spNN = \
        get_jn_derivs(sys, efn, v, s_spNN, dz)

        djz_smNN_defn_smNN, djz_smNN_defn_s, djz_smNN_dv_smNN, djz_smNN_dv_s = \
        get_jn_derivs(sys, efn, v, smNN_s, dzm1)

        # compute the derivatives of fn
        defn_smNN = - djz_smNN_defn_smNN / dzbar
        dv_smNN = - djz_smNN_dv_smNN / dzbar

        defn_smN = - djy_smN_defn_smN / dybar
        dv_smN = - djy_smN_dv_smN / dybar

        defn_sm1 = - djx_sm1_defn_sm1 / dxbar
        dv_sm1 = - djx_sm1_dv_sm1 / dxbar

        defn_s = (djx_s_defn_s - djx_sm1_defn_s) / dxbar + \
                 (djy_s_defn_s - djy_smN_defn_s) / dybar + \
                 (djz_s_defn_s - djz_smNN_defn_s) / dzbar - dr_defn_s[sites]
        defp_s = - dr_defp_s[sites]
        dv_s = (djx_s_dv_s - djx_sm1_dv_s) / dxbar + \
               (djy_s_dv_s - djy_smN_dv_s) / dybar + \
               (djz_s_dv_s - djz_smNN_dv_s) / dzbar - dr_dv_s[sites]

        defn_sp1 = djx_s_defn_sp1 / dxbar
        dv_sp1 = djx_s_dv_sp1 / dxbar

        defn_spN = djy_s_defn_spN / dybar
        dv_spN = djy_s_dv_spN / dybar

        defn_spNN = djz_s_defn_spNN / dzbar
        dv_spNN = djz_s_dv_spNN / dzbar


        return defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, \
               defn_s, defp_s, dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN,\
               defn_spNN, dv_spNN

    def fp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                       dx, dxm1, dy, dym1, dz, dzm1, sites):

        # lattice distances
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        # pairs of sites in the x-direction (always defined like this)
        sm1_s = [i for i in zip(sites - 1, sites)]
        s_sp1 = [i for i in zip(sites, sites + 1)]

        # currents derivatives
        djx_s_defp_s, djx_s_defp_sp1, djx_s_dv_s, djx_s_dv_sp1 = \
        get_jp_derivs(sys, efp, v, s_sp1, dx)

        djx_sm1_defp_sm1, djx_sm1_defp_s, djx_sm1_dv_sm1, djx_sm1_dv_s = \
        get_jp_derivs(sys, efp, v, sm1_s, dxm1)

        djy_s_defp_s, djy_s_defp_spN, djy_s_dv_s, djy_s_dv_spN = \
        get_jp_derivs(sys, efp, v, s_spN, dy)

        djy_smN_defp_smN, djy_smN_defp_s, djy_smN_dv_smN, djy_smN_dv_s = \
        get_jp_derivs(sys, efp, v, smN_s, dym1)

        djz_s_defp_s, djz_s_defp_spNN, djz_s_dv_s, djz_s_dv_spNN = \
        get_jp_derivs(sys, efp, v, s_spNN, dz)

        djz_smNN_defp_smNN, djz_smNN_defp_s, djz_smNN_dv_smNN, djz_smNN_dv_s = \
        get_jp_derivs(sys, efp, v, smNN_s, dzm1)

        # compute the derivatives of fp
        defp_smNN = - djz_smNN_defp_smNN / dzbar
        dv_smNN = - djz_smNN_dv_smNN / dzbar

        defp_smN = - djy_smN_defp_smN / dybar
        dv_smN = - djy_smN_dv_smN / dybar

        defp_sm1 = - djx_sm1_defp_sm1 / dxbar
        dv_sm1 = - djx_sm1_dv_sm1 / dxbar

        defn_s = dr_defn_s[sites]
        defp_s = (djx_s_defp_s - djx_sm1_defp_s) / dxbar + \
                 (djy_s_defp_s - djy_smN_defp_s) / dybar + \
                 (djz_s_defp_s - djz_smNN_defp_s) / dzbar + dr_defp_s[sites]
        dv_s = (djx_s_dv_s - djx_sm1_dv_s) / dxbar + \
               (djy_s_dv_s - djy_smN_dv_s) / dybar + \
               (djz_s_dv_s - djz_smNN_dv_s) / dzbar + dr_dv_s[sites]

        defp_sp1 = djx_s_defp_sp1 / dxbar
        dv_sp1 = djx_s_dv_sp1 / dxbar

        defp_spN = djy_s_defp_spN / dybar
        dv_spN = djy_s_dv_spN / dybar

        defp_spNN = djz_s_defp_spNN / dzbar
        dv_spNN = djz_s_dv_spNN / dzbar

        return defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1,\
               defn_s, defp_s, dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN,\
               defp_spNN, dv_spNN


    def fv_derivatives(sys, dx, dxm1, dy, dym1, dz, dzm1, sites):

        # lattice distances
        dxbar = (dx + dxm1) / 2
        dybar = (dy + dym1) / 2
        dzbar = (dz + dzm1) / 2

        # compute the derivatives
        dvmNN = -1./(dzm1 * dzbar)
        dvmN = -1./(dym1 * dybar)
        dvm1 = -1./(dxm1 * dxbar)
        dv = 2./(dx * dxm1) + 2./(dy * dym1)\
           + 2./(dz * dzm1) - drho_dv_s[sites]
        defn = - drho_defn_s[sites]
        defp = - drho_defp_s[sites]
        dvp1 = -1./(dx * dxbar)
        dvpN = -1./(dy * dybar)
        dvpNN = -1./(dz * dzbar)
        
        return dvmNN, dvmN, dvm1, defn, defp, dv, dvp1, dvpN, dvpNN

    def bn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN,\
                       dx, dxm1, dy, dym1, dz, dzm1, sites):

        # lattice distances
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        # pairs of sites in the x-direction (always defined like this)
        sm1_s = [i for i in zip(sites - 1, sites)]

        # compute the currents derivatives
        djnx_sm1_defn_sm1, djnx_sm1_defn_s, djnx_sm1_dv_sm1, djnx_sm1_dv_s =\
        get_jn_derivs(sys, efn, v, sm1_s, dxm1)

        djny_s_defn_s, djny_s_defn_spN, djny_s_dv_s, djny_s_dv_spN = \
        get_jn_derivs(sys, efn, v, s_spN, dy)

        djny_smN_defn_smN, djny_smN_defn_s, djny_smN_dv_smN, djny_smN_dv_s = \
        get_jn_derivs(sys, efn, v, smN_s, dym1)

        djnz_s_defn_s, djnz_s_defn_spNN, djnz_s_dv_s, djnz_s_dv_spNN = \
        get_jn_derivs(sys, efn, v, s_spNN, dz)

        djnz_smNN_defn_smNN, djnz_smNN_defn_s, djnz_smNN_dv_smNN, djnz_smNN_dv_s = \
        get_jn_derivs(sys, efn, v, smNN_s, dzm1)

        # compute bn derivatives
        defn_smNN = dxbar/dzbar * djnz_smNN_defn_smNN
        dv_smNN = dxbar/dzbar * djnz_smNN_dv_smNN

        defn_smN = dxbar/dybar * djny_smN_defn_smN
        dv_smN = dxbar/dybar * djny_smN_dv_smN
        
        defn_sm1 = djnx_sm1_defn_sm1
        dv_sm1 = djnx_sm1_dv_sm1

        defn_s = djnx_sm1_defn_s + dxbar * (dr_defn_s[sites]\
                 - (djny_s_defn_s - djny_smN_defn_s) / dybar\
                 - (djny_s_defn_s - djnz_smNN_defn_s) / dzbar) + sys.Scn[1] * n[sites]
        defp_s = dxbar * dr_defp_s[sites]
        dv_s = djnx_sm1_dv_s + dxbar * (dr_dv_s[sites]\
               - (djny_s_dv_s  - djny_smN_dv_s) / dybar\
               - (djnz_s_dv_s  - djnz_smNN_dv_s) / dzbar) + sys.Scn[1] * n[sites]

        defn_spN = - dxbar/dybar * djny_s_defn_spN
        dv_spN = - dxbar/dybar * djny_s_dv_spN

        defn_spNN = - dxbar/dzbar * djnz_s_defn_spNN
        dv_spNN = - dxbar/dzbar * djnz_s_dv_spNN
        
        return defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s,\
               defp_s, dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN

    def bp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                       dx, dxm1, dy, dym1, dz, dzm1, sites):

        # lattice distances
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        # pairs of sites in the x-direction (always defined like this)
        sm1_s = [i for i in zip(sites - 1, sites)]

        # compute the currents derivatives
        djpx_sm1_defp_sm1, djpx_sm1_defp_s, djpx_sm1_dv_sm1, djpx_sm1_dv_s =\
        get_jp_derivs(sys, efp, v, sm1_s, dx)

        djpy_s_defp_s, djpy_s_defp_spN, djpy_s_dv_s, djpy_s_dv_spN = \
        get_jp_derivs(sys, efp, v, s_spN, dy)

        djpy_smN_defp_smN, djpy_smN_defp_s, djpy_smN_dv_smN, djpy_smN_dv_s = \
        get_jp_derivs(sys, efp, v, smN_s, dym1)

        djpz_s_defp_s, djpz_s_defp_spNN, djpz_s_dv_s, djpz_s_dv_spNN = \
        get_jp_derivs(sys, efp, v, s_spNN, dz)

        djpz_smNN_defp_smNN, djpz_smNN_defp_s, djpz_smNN_dv_smNN, djpz_smNN_dv_s = \
        get_jp_derivs(sys, efp, v, smNN_s, dzm1)

        # compute bn derivatives
        defp_smNN = dxbar/dzbar * djpz_smNN_defp_smNN
        dv_smNN = dxbar/dzbar * djpz_smNN_dv_smNN

        defp_smN = dxbar/dybar * djpy_smN_defp_smN
        dv_smN = dxbar/dybar * djpy_smN_dv_smN

        defp_sm1 = djpx_sm1_defp_sm1
        dv_sm1 = djpx_sm1_dv_sm1

        defn_s = - dxbar * dr_defn_s[sites]
        defp_s = djpx_sm1_defp_s + dxbar * (-dr_defp_s[sites]\
                 - (djpy_s_defp_s - djpy_smN_defp_s) / dybar \
                 - (djpz_s_defp_s - djpz_smNN_defp_s) / dzbar) - sys.Scp[1] * p[sites]
        dv_s = djpx_sm1_dv_s + dxbar * (-dr_dv_s[sites] \
               - (djpy_s_dv_s  - djpy_smN_dv_s) / dybar \
               - (djpz_s_dv_s  - djpz_smNN_dv_s) / dzbar) + sys.Scp[1] * p[sites]

        defp_spN = - dxbar/dybar * djpy_s_defp_spN
        dv_spN = - dxbar/dybar * djpy_s_dv_spN

        defp_spNN = - dxbar/dzbar * djpz_s_defp_spNN
        dv_spNN = - dxbar/dzbar * djpz_s_dv_spNN

        return defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s,\
               defp_s, dv_s, defp_spN, dv_spN, defp_spNN, dv_spNN

    def bv_derivatives(sites):
        dbv_rows = [3*s+2 for s in sites]
        dbv_cols = [3*s+2 for s in sites]
        dbv_data = [1 for s in sites] # dv_s = 0

        global rows, columns, data
        rows += dbv_rows
        columns += dbv_cols
        data += dbv_data

    ###########################################################################
    #     inside the system: 0 < i < Nx-1,  0 < j < Ny-1, 0 < k < Nz-1        #
    ###########################################################################
    # We compute fn, fp, fv derivatives. Those functions are only defined on the
    # inner part of the system. All the edges containing boundary conditions.

    # list of the sites inside the system
    sites = [i + j*Nx + k*Nx*Ny for k in range(1,Nz-1) 
                                for j in range(1,Ny-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], (Ny-2)*(Nz-2))
    dy = np.repeat(sys.dy[1:], (Nx-2)*(Nz-2)) 
    dz = np.repeat(sys.dz[1:], (Nx-2)*(Ny-2))
    dxm1 = np.tile(sys.dx[:-1], (Ny-2)*(Nz-2))
    dym1 = np.repeat(sys.dy[:-1], (Nx-2)*(Nz-2))
    dzm1 = np.repeat(sys.dz[:-1], (Nx-2)*(Ny-2))

    # gather all relevant pairs of sites to compute the currents derivatives
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    sm1_s = [sites - 1, sites]
    s_sp1 = [sites, sites + 1]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    #------------------------ fn derivatives ----------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    fn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = np.reshape(np.repeat(3*sites, 15), (len(sites), 15)).tolist()

    dfn_cols = zip(3*(sites-Nx*Ny), 3*(sites-Nx*Ny)+2, 3*(sites-Nx),\
                   3*(sites-Nx)+2, 3*(sites-1), 3*(sites-1)+2, 3*sites,\
                   3*sites+1, 3*sites+2, 3*(sites+1), 3*(sites+1)+2, 3*(sites+Nx),\
                   3*(sites+Nx)+2, 3*(sites+Nx*Ny), 3*(sites+Nx*Ny)+2)

    dfn_data = zip(defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, \
                   defn_s, defp_s, dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN,\
                   defn_spNN, dv_spNN)

    update(dfn_rows, dfn_cols, dfn_data)

    #------------------------ fp derivatives ----------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    fp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = np.reshape(np.repeat(3*sites+1, 15), (len(sites), 15)).tolist()

    dfp_cols = zip(3*(sites-Nx*Ny)+1, 3*(sites-Nx*Ny)+2, 3*(sites-Nx)+1, 3*(sites-Nx)+2,\
                 3*(sites-1)+1, 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,\
                 3*(sites+1)+1, 3*(sites+1)+2, 3*(sites+Nx)+1, 3*(sites+Nx)+2,\
                 3*(sites+Nx*Ny)+1, 3*(sites+Nx*Ny)+2)

    dfp_data = zip(defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1,\
                   defn_s, defp_s, dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN,\
                   defp_spNN, dv_spNN)

    update(dfp_rows, dfp_cols, dfp_data)

    #---------------- fv derivatives inside the system ------------------------
    # compute the derivatives
    dvmNN, dvmN, dvm1, defn, defp, dv, dvp1, dvpN, dvpNN = \
    fv_derivatives(sys, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = np.reshape(np.repeat(3*sites+2, 9), (len(sites), 9)).tolist()

    dfv_cols = zip(3*(sites-Nx*Ny)+2, 3*(sites-Nx)+2, 3*(sites-1)+2, 3*sites,\
                   3*sites+1, 3*sites+2, 3*(sites+1)+2, 3*(sites+Nx)+2, 3*(sites+Nx*Ny)+2)

    dfv_data = zip(dvmNN, dvmN, dvm1, defn, defp, dv, dvp1, dvpN, dvpNN)

    update(dfv_rows, dfv_cols, dfv_data)


    ###########################################################################
    #        left boundary: i = 0, 0 <= j <= Ny-1, 0 <= k <= Nz-1             #
    ###########################################################################
    # We compute an, ap, av derivatives. Those functions are only defined on the
    # left boundary of the system.

    # list of the sites on the left side
    sites = [j*Nx + k*Nx*Ny for k in range(Nz) for j in range(Ny)]
    sites = np.asarray(sites)

    #-------------------------- an derivatives --------------------------------
    s_sp1 = [sites, sites + 1]
    defn_s, defn_sp1, dv_s, dv_sp1 = get_jn_derivs(sys, efn, v, s_sp1)

    defn_s -= sys.Scn[0] * n[sites]
    dv_s -= sys.Scn[0] * n[sites]

    # update the sparse matrix row and columns
    dan_rows = np.reshape(np.repeat(3*sites, 4), (len(sites), 4)).tolist()

    dan_cols = zip(3*sites, 3*sites+2, 3*(sites+1), 3*(sites+1)+2)

    dan_data = zip(defn_s, dv_s, defn_sp1, dv_sp1)

    update(dan_rows, dan_cols, dan_data)

    #-------------------------- ap derivatives --------------------------------
    defp_s, defp_sp1, dv_s, dv_sp1 = get_jp_derivs(sys, efp, v, s_sp1)

    defp_s += sys.Scp[0] * p[sites]
    dv_s -= sys.Scp[0] * p[sites]

    # update the sparse matrix row and columns
    dap_rows = np.reshape(np.repeat(3*sites+1, 4), (len(sites), 4)).tolist()

    dap_cols = zip(3*sites+1, 3*sites+2, 3*(sites+1)+1, 3*(sites+1)+2)

    dap_data = zip(defp_s, dv_s, defp_sp1, dv_sp1)
    
    update(dap_rows, dap_cols, dap_data)

    #-------------------------- av derivatives --------------------------------
    dav_rows = (3*sites+2).tolist()
    dav_cols = (3*sites+2).tolist()
    dav_data = [1 for s in sites]

    rows += dav_rows
    columns += dav_cols
    data += dav_data

    ###########################################################################
    #         right boundary: i = Nx-1, 0 < j < Ny-1, 0 < k < Nz-1            #
    ###########################################################################
    # We compute bn, bp, bv derivatives. Those functions are only defined on the
    # right boundary of the system.

    # list of the sites on the right side
    sites = [Nx-1 + j*Nx + k*Nx*Ny for k in range(1,Nz-1) for j in range(1,Ny-1)]
    sites = np.asarray(sites)

    # dxbar and dybar
    dx = np.tile(sys.dx[-1], (Ny-2)*(Nz-2))
    dxm1 = np.tile(sys.dx[-1], (Ny-2)*(Nz-2))
    dy = np.repeat(sys.dy[1:], Nz-2)
    dym1 = np.repeat(sys.dy[:-1], Nz-2)
    dz = np.repeat(sys.dz[1:], Ny-2)
    dzm1 = np.repeat(sys.dz[:-1], Ny-2)

    # compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    #-------------------------- bn derivatives --------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    bn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbn_rows = np.reshape(np.repeat(3*sites, 13), (len(sites), 13)).tolist()

    dbn_cols = zip(3*(sites-Nx*Ny), 3*(sites-Nx*Ny)+2, 3*(sites-Nx),
                   3*(sites-Nx)+2, 3*(sites-1), 3*(sites-1)+2, 3*sites,\
                   3*sites+1, 3*sites+2, 3*(sites+Nx), 3*(sites+Nx)+2,\
                   3*(sites+Nx*Ny), 3*(sites+Nx*Ny)+2)

    dbn_data = zip(defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1,\
                   defn_s, defp_s, dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN)

    update(dbn_rows, dbn_cols, dbn_data)

    #-------------------------- bp derivatives --------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    bp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbp_rows = np.reshape(np.repeat(3*sites+1, 13), (len(sites), 13)).tolist()

    dbp_cols = zip(3*(sites-Nx*Ny)+1, 3*(sites-Nx*Ny)+2, 3*(sites-Nx)+1,
                   3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2, 3*sites,\
                   3*sites+1, 3*sites+2, 3*(sites+Nx)+1, 3*(sites+Nx)+2,\
                   3*(sites+Nx*Ny)+1, 3*(sites+Nx*Ny)+2)

    dbp_data = zip(defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1,\
                   defn_s, defp_s, dv_s, defp_spN, dv_spN, defp_spNN, dv_spNN)

    update(dbp_rows, dbp_cols, dbp_data)

    #-------------------------- bv derivatives --------------------------------
    bv_derivatives(sites)

    ###########################################################################
    #           right boundary: i = Nx-1, j = Ny-1, 0 < k < Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + (Ny-1)*Nx + k*Nx*Ny for k in range(1,Nz-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[-1], Nz-2)
    dxm1 = np.tile(sys.dx[-1], Nz-2)
    dy = np.repeat((sys.dy[0] + sys.dy[-1]) / 2., Nz-2)
    dym1 = np.repeat(sys.dy[-1], Nz-2) 
    dz = sys.dz[1:]
    dzm1 = sys.dz[:-1]

    # compute the currents
    smNN_s =[sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites - Nx*(Ny-1)]
    s_spNN =[sites, sites + Nx*Ny]

    #-------------------------- bn derivatives --------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    bn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbn_rows = np.reshape(np.repeat(3*sites, 13), (len(sites), 13)).tolist()

    dbn_cols = zip(3*(sites-Nx*Ny), 3*(sites-Nx*Ny)+2, 3*(sites-Nx*(Ny-1)),
                   3*(sites-Nx*(Ny-1))+2, 3*(sites-Nx), 3*(sites-Nx)+2,\
                   3*(sites-1), 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,\
                   3*(sites+Nx*Ny), 3*(sites+Nx*Ny)+2)

    dbn_data = zip(defn_smNN, dv_smNN, defn_spN, dv_spN, defn_smN, dv_smN,\
                   defn_sm1, dv_sm1, defn_s, defp_s, dv_s, defn_spNN, dv_spNN)

    update(dbn_rows, dbn_cols, dbn_data)

    #-------------------------- bp derivatives --------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    bp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbp_rows = np.reshape(np.repeat(3*sites+1, 13), (len(sites), 13)).tolist()

    dbp_cols = zip(3*(sites-Nx*Ny)+1, 3*(sites-Nx*Ny)+2, 3*(sites-Nx*(Ny-1))+1,
                   3*(sites-Nx*(Ny-1))+2, 3*(sites-Nx)+1, 3*(sites-Nx)+2,\
                   3*(sites-1)+1, 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,\
                   3*(sites+Nx*Ny)+1, 3*(sites+Nx*Ny)+2)

    dbp_data = zip(defp_smNN, dv_smNN, defp_spN, dv_spN, defp_smN, dv_smN,\
                   defp_sm1, dv_sm1, defn_s, defp_s, dv_s, defp_spNN, dv_spNN)

    update(dbp_rows, dbp_cols, dbp_data)

    #-------------------------- bv derivatives --------------------------------
    bv_derivatives(sites)

    ###########################################################################
    #              right boundary: i = Nx-1, j = 0, 0 < k < Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + k*Nx*Ny for k in range(1,Nz-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = np.repeat(sys.dy[-1], Nz-2)
    dym1 =  np.repeat((sys.dy[0] + sys.dy[-1]) / 2., Nz-2)
    dz = sys.dz[1:]
    dzm1 = sys.dz[:-1]

    # compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites + Nx*(Ny-1), sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    #-------------------------- bn derivatives --------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    bn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbn_rows = np.reshape(np.repeat(3*sites, 13), (len(sites), 13)).tolist()

    dbn_cols = zip(3*(sites-Nx*Ny), 3*(sites-Nx*Ny)+2, 3*(sites-1),
                   3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2, 3*(sites+Nx),\
                   3*(sites+Nx)+2, 3*(sites+Nx*(Ny-1)), 3*(sites+Nx*(Ny-1))+2,\
                   3*(sites+Nx*Ny), 3*(sites+Nx*Ny)+2)

    dbn_data = zip(defn_smNN, dv_smNN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defn_spN, dv_spN, defn_smN, dv_smN, defn_spNN, dv_spNN)

    update(dbn_rows, dbn_cols, dbn_data)

    #-------------------------- bp derivatives --------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    bp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbp_rows = np.reshape(np.repeat(3*sites+1, 13), (len(sites), 13)).tolist()

    dbp_cols = zip(3*(sites-Nx*Ny)+1, 3*(sites-Nx*Ny)+2, 3*(sites-1)+1,\
                   3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2, 3*(sites+Nx)+1,\
                   3*(sites+Nx)+2, 3*(sites+Nx*(Ny-1))+1, 3*(sites+Nx*(Ny-1))+2,\
                   3*(sites+Nx*Ny)+1, 3*(sites+Nx*Ny)+2)

    dbp_data = zip(defp_smNN, dv_smNN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defp_spN, dv_spN, defp_smN, dv_smN, defp_spNN, dv_spNN)

    update(dbp_rows, dbp_cols, dbp_data)

    #-------------------------- bv derivatives --------------------------------
    bv_derivatives(sites)

    ###########################################################################
    #           right boundary: i = Nx-1, 0 < j < Ny-1, k = Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + j*Nx + (Nz-1)*Nx*Ny for j in range(1,Ny-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[-1], Ny-2)
    dxm1 = np.tile(sys.dx[-1], Ny-2)
    dy = sys.dy[1:]
    dym1 = sys.dy[:-1]
    dz = np.repeat((sys.dz[-1] + sys.dz[0])/2., Ny-2)
    dzm1 = np.repeat(sys.dz[-1], Ny-2)

    # compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites - Nx*Ny*(Nz-1)]

    #-------------------------- bn derivatives --------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    bn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbn_rows = np.reshape(np.repeat(3*sites, 13), (len(sites), 13)).tolist()

    dbn_cols = zip(3*(sites-Nx*Ny*(Nz-1)), 3*(sites-Nx*Ny*(Nz-1))+2, 3*(sites-Nx*Ny),\
                 3*(sites-Nx*Ny)+2, 3*(sites-Nx), 3*(sites-Nx)+2, 3*(sites-1),
                 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2, 3*(sites+Nx),\
                 3*(sites+Nx)+2)

    dbn_data = zip(defn_spNN, dv_spNN, defn_smNN, dv_smNN, defn_smN, dv_smN,\
                   defn_sm1, dv_sm1, defn_s, defp_s, dv_s, defn_spN, dv_spN)

    update(dbn_rows, dbn_cols, dbn_data)

    #-------------------------- bp derivatives --------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    bp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbp_rows = np.reshape(np.repeat(3*sites+1, 13), (len(sites), 13)).tolist()

    dbp_cols = zip(3*(sites-Nx*Ny*(Nz-1))+1, 3*(sites-Nx*Ny*(Nz-1))+2,
                   3*(sites-Nx*Ny)+1, 3*(sites-Nx*Ny)+2, 3*(sites-Nx)+1,\
                   3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2, 3*sites,\
                   3*sites+1, 3*sites+2, 3*(sites+Nx)+1, 3*(sites+Nx)+2)

    dbp_data = zip(defp_spNN, dv_spNN, defp_smNN, dv_smNN, defp_smN, dv_smN,\
                   defp_sm1, dv_sm1, defn_s, defp_s, dv_s, defp_spN, dv_spN)

    update(dbp_rows, dbp_cols, dbp_data)

    #-------------------------- bv derivatives --------------------------------
    bv_derivatives(sites)

    ###########################################################################
    #              right boundary: i = Nx-1, 0 < j < Ny-1, k = 0              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + j*Nx for j in range(1,Ny-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = sys.dy[1:]
    dym1 = sys.dy[:-1]
    dz = np.repeat(sys.dz[0], Ny-2)
    dzm1 = np.repeat((sys.dz[-1] + sys.dz[0])/2., Ny-2)

    # compute the currents
    smNN_s = [sites + Nx*Ny*(Nz-1), sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    #-------------------------- bn derivatives --------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    bn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbn_rows = np.reshape(np.repeat(3*sites, 13), (len(sites), 13)).tolist()

    dbn_cols = zip(3*(sites-Nx), 3*(sites-Nx)+2, 3*(sites-1), 3*(sites-1)+2,
                   3*sites, 3*sites+1, 3*sites+2, 3*(sites+Nx), 3*(sites+Nx)+2,\
                   3*(sites+Nx*Ny), 3*(sites+Nx*Ny)+2, 3*(sites+Nx*Ny*(Nz-1)),\
                   3*(sites+Nx*Ny*(Nz-1))+2)

    dbn_data = zip(defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defn_spN, dv_spN, defn_spNN, dv_spNN, defn_smNN, dv_smNN)

    update(dbn_rows, dbn_cols, dbn_data)

    #-------------------------- bp derivatives --------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    bp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbp_rows = np.reshape(np.repeat(3*sites+1, 13), (len(sites), 13)).tolist()

    dbp_cols = zip(3*(sites-Nx)+1, 3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2,
                   3*sites, 3*sites+1, 3*sites+2, 3*(sites+Nx)+1, 3*(sites+Nx)+2,\
                   3*(sites+Nx*Ny)+1, 3*(sites+Nx*Ny)+2,\
                   3*(sites+Nx*Ny*(Nz-1))+1, 3*(sites+Nx*Ny*(Nz-1))+2)

    dbp_data = zip(defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defp_spN, dv_spN, defp_spNN, dv_spNN, defp_smNN, dv_smNN)

    update(dbp_rows, dbp_cols, dbp_data)

    #-------------------------- bv derivatives --------------------------------
    bv_derivatives(sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = Ny-1, k = 0              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + (Ny-1)*Nx]
    sites = np.asarray(sites)

    # lattice distances
    dx = sys.dx[-1]
    dxm1 = sys.dx[-1]
    dy = (sys.dy[0] + sys.dy[-1])/2.
    dym1 = sys.dy[-1]
    dz = sys.dz[0]
    dzm1 = (sys.dz[-1] + sys.dz[0])/2.

    # compute the currents
    smNN_s = [i for i in zip(sites + Nx*Ny*(Nz-1), sites)]
    smN_s = [i for i in zip(sites - Nx, sites)]
    s_spN = [i for i in zip(sites, sites - Nx*(Ny-1))]
    s_spNN = [i for i in zip(sites, sites + Nx*Ny)]

    #-------------------------- bn derivatives --------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    bn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbn_rows = np.reshape(np.repeat(3*sites, 13), (len(sites), 13)).tolist()

    dbn_cols = zip(3*(sites-Nx*(Ny-1)), 3*(sites-Nx*(Ny-1))+2, 3*(sites-Nx),
                   3*(sites-Nx)+2, 3*(sites-1), 3*(sites-1)+2, 3*sites,\
                   3*sites+1, 3*sites+2, 3*(sites+Nx*Ny), 3*(sites+Nx*Ny)+2,\
                   3*(sites+Nx*Ny*(Nz-1)), 3*(sites+Nx*Ny*(Nz-1))+2)

    dbn_data = zip(defn_spN, dv_spN, defn_smN, dv_smN, defn_sm1, dv_sm1,\
                   defn_s, defp_s, dv_s, defn_spNN, dv_spNN, defn_smNN, dv_smNN)

    update(dbn_rows, dbn_cols, dbn_data)

    #-------------------------- bp derivatives --------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    bp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbp_rows = np.reshape(np.repeat(3*sites+1, 13), (len(sites), 13)).tolist()

    dbp_cols = zip(3*(sites-Nx*(Ny-1))+1, 3*(sites-Nx*(Ny-1))+2, 3*(sites-Nx)+1,
                   3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2, 3*sites,\
                   3*sites+1, 3*sites+2, 3*(sites+Nx*Ny)+1,\
                   3*(sites+Nx*Ny)+2, 3*(sites+Nx*Ny*(Nz-1))+1, 3*(sites+Nx*Ny*(Nz-1))+2)

    dbp_data = zip(defp_spN, dv_spN, defp_smN, dv_smN, defp_sm1, dv_sm1,\
                   defn_s, defp_s, dv_s, defp_spNN, dv_spNN, defp_smNN, dv_smNN)

    update(dbp_rows, dbp_cols, dbp_data)

    #-------------------------- bv derivatives --------------------------------
    bv_derivatives(sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = Ny-1, k = Nz-1           #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + (Ny-1)*Nx + (Nz-1)*Nx*Ny]
    sites = np.asarray(sites)

    # lattice distances
    dy = (sys.dy[0] + sys.dy[-1])/2.
    dym1 = sys.dy[-1]
    dz = (sys.dz[-1] + sys.dz[0])/2.
    dzm1 = sys.dz[-1]

    # compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites - Nx*(Ny-1)]
    s_spNN = [sites, sites - Nx*Ny*(Nz-1)]

    #-------------------------- bn derivatives --------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    bn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbn_rows = np.reshape(np.repeat(3*sites, 13), (len(sites), 13)).tolist()

    dbn_cols = zip(3*(sites-Nx*Ny*(Nz-1)), 3*(sites-Nx*Ny*(Nz-1))+2, 3*(sites-Nx*Ny),\
                 3*(sites-Nx*Ny)+2, 3*(sites-Nx*(Ny-1)), 3*(sites-Nx*(Ny-1))+2,\
                 3*(sites-Nx), 3*(sites-Nx)+2, 3*(sites-1), 3*(sites-1)+2, 3*sites, 3*sites+1,
                 3*sites+2)

    dbn_data = zip(defn_spNN, dv_spNN, defn_smNN, dv_smNN, defn_spN, dv_spN,\
                   defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s)

    update(dbn_rows, dbn_cols, dbn_data)

    #-------------------------- bp derivatives --------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    bp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbp_rows = np.reshape(np.repeat(3*sites+1, 13), (len(sites), 13)).tolist()

    dbp_cols = zip(3*(sites-Nx*Ny*(Nz-1))+1, 3*(sites-Nx*Ny*(Nz-1))+2, 3*(sites-Nx*Ny)+1,\
                 3*(sites-Nx*Ny)+2, 3*(sites-Nx*(Ny-1))+1, 3*(sites-Nx*(Ny-1))+2, \
                 3*(sites-Nx)+1, 3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2, 3*sites,
                 3*sites+1, 3*sites+2)

    dbp_data = zip(defp_spNN, dv_spNN, defp_smNN, dv_smNN, defp_spN, dv_spN,\
                   defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s)

    update(dbp_rows, dbp_cols, dbp_data)

    #-------------------------- bv derivatives --------------------------------
    bv_derivatives(sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = 0, k = Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + (Nz-1)*Nx*Ny]
    sites = np.asarray(sites)

    # lattice distances
    dy = sys.dy[0]
    dym1 = (sys.dy[0] + sys.dy[-1])/2.
    dz = (sys.dz[-1] + sys.dz[0])/2.
    dzm1 = sys.dz[-1]

    # compute the currents
    smNN_s =[sites - Nx*Ny, sites]
    smN_s = [sites + Nx*(Ny-1), sites]
    s_spN = [sites, sites + Nx]
    s_spNN =[sites, sites - Nx*Ny*(Nz-1)]

    #-------------------------- bn derivatives --------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    bn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbn_rows = np.reshape(np.repeat(3*sites, 13), (len(sites), 13)).tolist()

    dbn_cols = zip(3*(sites-Nx*Ny*(Nz-1)), 3*(sites-Nx*Ny*(Nz-1))+2, 3*(sites-Nx*Ny),\
                 3*(sites-Nx*Ny)+2, 3*(sites-1), 3*(sites-1)+2, 3*sites,
                 3*sites+1, 3*sites+2, 3*(sites+Nx), 3*(sites+Nx)+2,\
                 3*(sites+Nx*(Ny-1)), 3*(sites+Nx*(Ny-1))+2)

    dbn_data = zip(defn_spNN, dv_spNN, defn_smNN, dv_smNN, defn_sm1, dv_sm1,\
                   defn_s, defp_s, dv_s, defn_spN, dv_spN, defn_smN, dv_smN)

    update(dbn_rows, dbn_cols, dbn_data)

    #-------------------------- bp derivatives --------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    bp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbp_rows = np.reshape(np.repeat(3*sites+1, 13), (len(sites), 13)).tolist()

    dbp_cols = zip(3*(sites-Nx*Ny*(Nz-1))+1, 3*(sites-Nx*Ny*(Nz-1))+2, 3*(sites-Nx*Ny)+1,\
                   3*(sites-Nx*Ny)+2, 3*(sites-1)+1, 3*(sites-1)+2, 3*sites,\
                   3*sites+1, 3*sites+2, 3*(sites+Nx)+1, 3*(sites+Nx)+2,\
                   3*(sites+Nx*(Ny-1))+1, 3*(sites+Nx*(Ny-1))+2)

    dbp_data = zip(defp_spNN, dv_spNN, defp_smNN, dv_smNN, defp_sm1, dv_sm1,\
                   defn_s, defp_s, dv_s, defp_spN, dv_spN, defp_smN, dv_smN)

    update(dbp_rows, dbp_cols, dbp_data)

    #-------------------------- bv derivatives --------------------------------
    bv_derivatives(sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = 0, k = 0                 #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1]
    sites = np.asarray(sites)

    # lattice distances
    dy = sys.dy[0]
    dym1 = (sys.dy[0] + sys.dy[-1])/2.
    dz = sys.dz[0]
    dzm1 = (sys.dz[-1] + sys.dz[0])/2.

    # compute the currents
    smNN_s =[sites + Nx*Ny*(Nz-1), sites]
    smN_s = [sites + Nx*(Ny-1), sites]
    s_spN = [sites, sites + Nx]
    s_spNN =[sites, sites + Nx*Ny]

    #-------------------------- bn derivatives --------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    bn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbn_rows = np.reshape(np.repeat(3*sites, 13), (len(sites), 13)).tolist()

    dbn_cols = zip(3*(sites-1), 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,
                   3*(sites+Nx), 3*(sites+Nx)+2, 3*(sites+Nx*(Ny-1)),\
                   3*(sites+Nx*(Ny-1))+2, 3*(sites+Nx*Ny), 3*(sites+Nx*Ny)+2,\
                   3*(sites+Nx*Ny*(Nz-1)), 3*(sites+Nx*Ny*(Nz-1))+2)

    dbn_data = zip(defn_sm1, dv_sm1, defn_s, defp_s, dv_s, defn_spN, dv_spN,\
                   defn_smN, dv_smN, defn_spNN, dv_spNN, defn_smNN, dv_smNN)

    update(dbn_rows, dbn_cols, dbn_data)

    #-------------------------- bp derivatives --------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    bp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1,\
                   dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns
    dbp_rows = np.reshape(np.repeat(3*sites+1, 13), (len(sites), 13)).tolist()

    dbp_cols = zip(3*(sites-1)+1, 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,
                   3*(sites+Nx)+1, 3*(sites+Nx)+2, 3*(sites+Nx*(Ny-1))+1,\
                   3*(sites+Nx*(Ny-1))+2, 3*(sites+Nx*Ny)+1, 3*(sites+Nx*Ny)+2,\
                   3*(sites+Nx*Ny*(Nz-1))+1, 3*(sites+Nx*Ny*(Nz-1))+2)

    dbp_data = zip(defp_sm1, dv_sm1, defn_s, defp_s, dv_s, defp_spN, dv_spN,\
                   defp_smN, dv_smN, defp_spNN, dv_spNN, defp_smNN, dv_smNN)

    update(dbp_rows, dbp_cols, dbp_data)

    #-------------------------- bv derivatives --------------------------------
    bv_derivatives(sites)



    ###########################################################################
    #            faces between contacts: 0 < i < Nx-1, j or k fixed           #
    ###########################################################################
    # Here we focus on the faces between the contacts. There are 4 cases
    # (obviously).

    ###########################################################################
    #              z-face top: 0 < i < Nx-1, 0 < j < Ny-1, k = Nz-1           #
    ###########################################################################
    # list of the sites
    sites = [i + j*Nx + (Nz-1)*Nx*Ny for j in range(1,Ny-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny-2)
    dy = np.repeat(sys.dy[1:], Nx-2)
    dz = np.repeat((sys.dz[0] + sys.dz[-1])/2., (Nx-2)*(Ny-2))
    dxm1 = np.tile(sys.dx[:-1], Ny-2)
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dzm1 = np.repeat(sys.dz[-1], (Nx-2)*(Ny-2))

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites - Nx*Ny*(Nz-1)]

    #------------------------ fn derivatives ----------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    fn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = np.reshape(np.repeat(3*sites, 15), (len(sites), 15)).tolist()

    dfn_cols = zip(3*(sites-Nx*Ny*(Nz-1)), 3*(sites-Nx*Ny*(Nz-1))+2, 3*(sites-Nx*Ny), \
                 3*(sites-Nx*Ny)+2, 3*(sites-Nx), 3*(sites-Nx)+2, 3*(sites-1), \
                 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2, 3*(sites+1), 3*(sites+1)+2, 3*(sites+Nx),\
                 3*(sites+Nx)+2)

    dfn_data = zip(defn_spNN, dv_spNN, defn_smNN, dv_smNN, defn_smN, dv_smN,\
                   defn_sm1, dv_sm1, defn_s, defp_s, dv_s, defn_sp1, dv_sp1,\
                   defn_spN, dv_spN)

    update(dfn_rows, dfn_cols, dfn_data)

    #------------------------ fp derivatives ----------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    fp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = np.reshape(np.repeat(3*sites+1, 15), (len(sites), 15)).tolist()

    dfp_cols = zip(3*(sites-Nx*Ny*(Nz-1))+1, 3*(sites-Nx*Ny*(Nz-1))+2, 3*(sites-Nx*Ny)+1, \
                 3*(sites-Nx*Ny)+2, 3*(sites-Nx)+1, 3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2,\
                 3*sites, 3*sites+1, 3*sites+2, 3*(sites+1)+1, 3*(sites+1)+2, 3*(sites+Nx)+1, 3*(sites+Nx)+2)

    dfp_data = zip(defp_spNN, dv_spNN, defp_smNN, dv_smNN, defp_smN, dv_smN, \
                   defp_sm1, dv_sm1, defn_s, defp_s, dv_s, defp_sp1, dv_sp1,\
                   defp_spN, dv_spN)

    update(dfp_rows, dfp_cols, dfp_data)

    #---------------- fv derivatives inside the system ------------------------
    # compute the derivatives
    dvmNN, dvmN, dvm1, defn, defp, dv, dvp1, dvpN, dvpNN = \
    fv_derivatives(sys, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = np.reshape(np.repeat(3*sites+2, 9), (len(sites), 9)).tolist()

    dfv_cols = zip(3*(sites-Nx*Ny*(Nz-1))+2, 3*(sites-Nx*Ny)+2, 3*(sites-Nx)+2, 3*(sites-1)+2,\
                 3*sites, 3*sites+1, 3*sites+2, 3*(sites+1)+2, 3*(sites+Nx)+2)

    dfv_data = zip(dvpNN, dvmNN, dvmN, dvm1, defn, defp, dv, dvp1, dvpN)

    update(dfv_rows, dfv_cols, dfv_data)

    ###########################################################################
    #             z- face bottom: 0 < i < Nx-1, 0 < j < Ny-1, k = 0           #
    ###########################################################################
    # list of the sites
    sites = [i + j*Nx for j in range(1,Ny-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny-2)
    dy = np.repeat(sys.dy[1:], Nx-2)
    dz = np.repeat(sys.dz[0], (Nx-2)*(Ny-2))
    dxm1 = np.tile(sys.dx[:-1], Ny-2)
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dzm1 = np.repeat((sys.dz[0] + sys.dz[-1])/2., (Nx-2)*(Ny-2))

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites + Nx*Ny*(Nz-1), sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    #------------------------ fn derivatives ----------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    fn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = np.reshape(np.repeat(3*sites, 15), (len(sites), 15)).tolist()

    dfn_cols = zip(3*(sites-Nx), 3*(sites-Nx)+2, 3*(sites-1), 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,\
                 3*(sites+1), 3*(sites+1)+2, 3*(sites+Nx), 3*(sites+Nx)+2, 3*(sites+Nx*Ny),\
                 3*(sites+Nx*Ny)+2, 3*(sites+Nx*Ny*(Nz-1)), 3*(sites+Nx*Ny*(Nz-1))+2)

    dfn_data = zip(defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defn_sp1, dv_sp1, defn_spN, dv_spN, defn_spNN, dv_spNN, defn_smNN, dv_smNN)

    update(dfn_rows, dfn_cols, dfn_data)

    #------------------------ fp derivatives ----------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    fp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = np.reshape(np.repeat(3*sites+1, 15), (len(sites), 15)).tolist()

    dfp_cols = zip(3*(sites-Nx)+1, 3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,\
                 3*(sites+1)+1, 3*(sites+1)+2, 3*(sites+Nx)+1, 3*(sites+Nx)+2, 3*(sites+Nx*Ny)+1,\
                 3*(sites+Nx*Ny)+2, 3*(sites+Nx*Ny*(Nz-1))+1, 3*(sites+Nx*Ny*(Nz-1))+2)

    dfp_data = zip(defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defp_sp1, dv_sp1, defp_spN, dv_spN, defp_spNN, dv_spNN, defp_smNN, dv_smNN)

    update(dfp_rows, dfp_cols, dfp_data)

    #---------------- fv derivatives inside the system ------------------------
    # compute the derivatives
    dvmNN, dvmN, dvm1, defn, defp, dv, dvp1, dvpN, dvpNN = \
    fv_derivatives(sys, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = np.reshape(np.repeat(3*sites+2, 9), (len(sites), 9)).tolist()

    dfv_cols = zip(3*(sites-Nx)+2, 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2, 3*(sites+1)+2, 3*(sites+Nx)+2,\
                 3*(sites+Nx*Ny)+2, 3*(sites+Nx*Ny*(Nz-1))+2)

    dfv_data = zip(dvmN, dvm1, defn, defp, dv, dvp1, dvpN, dvpNN, dvmNN)

    update(dfv_rows, dfv_cols, dfv_data)

    ###########################################################################
    #            y-face front: 0 < i < Nx-1, j = 0, 0 < k < Nz-1              #
    ###########################################################################
    # list of the sites
    sites = [i + k*Nx*Ny for k in range(1,Nz-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Nz-2)
    dy = np.repeat(sys.dy[0], (Nx-2)*(Nz-2))
    dz = np.repeat(sys.dz[1:], (Nx-2))
    dxm1 = np.tile(sys.dx[:-1], Nz-2)
    dym1 = np.repeat((sys.dy[0] + sys.dy[-1])/2., (Nx-2)*(Nz-2))
    dzm1 = np.repeat(sys.dz[:-1], Nx-2)

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites + Nx*(Ny-1), sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    #------------------------ fn derivatives ----------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    fn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = np.reshape(np.repeat(3*sites, 15), (len(sites), 15)).tolist()

    dfn_cols = zip(3*(sites-Nx*Ny), 3*(sites-Nx*Ny)+2, 3*(sites-1), 3*(sites-1)+2, 3*sites, 3*sites+1,\
                 3*sites+2, 3*(sites+1), 3*(sites+1)+2, 3*(sites+Nx), 3*(sites+Nx)+2, 3*(sites+Nx*(Ny-1)),\
                 3*(sites+Nx*(Ny-1))+2, 3*(sites+Nx*Ny), 3*(sites+Nx*Ny)+2)

    dfn_data = zip(defn_smNN, dv_smNN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defn_sp1, dv_sp1, defn_spN, dv_spN, defn_smN, dv_smN,\
                   defn_spNN, dv_spNN)

    update(dfn_rows, dfn_cols, dfn_data)

    #------------------------ fp derivatives ----------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    fp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = np.reshape(np.repeat(3*sites+1, 15), (len(sites), 15)).tolist()

    dfp_cols = zip(3*(sites-Nx*Ny)+1, 3*(sites-Nx*Ny)+2, 3*(sites-1)+1, 3*(sites-1)+2, 3*sites, 3*sites+1,\
                 3*sites+2, 3*(sites+1)+1, 3*(sites+1)+2, 3*(sites+Nx)+1, 3*(sites+Nx)+2, 3*(sites+Nx*(Ny-1))+1,\
                 3*(sites+Nx*(Ny-1))+2, 3*(sites+Nx*Ny)+1, 3*(sites+Nx*Ny)+2)

    dfp_data = zip(defp_smNN, dv_smNN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defp_sp1, dv_sp1, defp_spN, dv_spN, defp_smN, dv_smN,\
                   defp_spNN, dv_spNN)

    update(dfp_rows, dfp_cols, dfp_data)

    #---------------- fv derivatives inside the system ------------------------
    # compute the derivatives
    dvmNN, dvmN, dvm1, defn, defp, dv, dvp1, dvpN, dvpNN = \
    fv_derivatives(sys, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = np.reshape(np.repeat(3*sites+2, 9), (len(sites), 9)).tolist()

    dfv_cols = zip(3*(sites-Nx*Ny)+2, 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2, 3*(sites+1)+2,\
                   3*(sites+Nx)+2, 3*(sites+Nx*(Ny-1))+2, 3*(sites+Nx*Ny)+2)

    dfv_data = zip(dvmNN, dvm1, defn, defp, dv, dvp1, dvpN, dvmN, dvpNN)

    update(dfv_rows, dfv_cols, dfv_data)

    ###########################################################################
    #            y-face back: 0 < i < Nx-1, j = Ny-1, 0 < k < Nz-1            #
    ###########################################################################
    # list of the sites
    sites = [i + (Ny-1)*Nx + k*Nx*Ny for k in range(1,Nz-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Nz-2)
    dy = np.repeat((sys.dy[0] + sys.dy[-1])/2., (Nx-2)*(Nz-2))
    dz = np.repeat(sys.dz[1:], Nx-2)
    dxm1 = np.tile(sys.dx[:-1], Nz-2)
    dym1 = np.repeat(sys.dy[0], (Nx-2)*(Nz-2))
    dzm1 = np.repeat(sys.dz[:-1], Nx-2)

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites - Nx*(Ny-1)]
    s_spNN = [sites, sites + Nx*Ny]

    #------------------------ fn derivatives ----------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    fn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = np.reshape(np.repeat(3*sites, 15), (len(sites), 15)).tolist()

    dfn_cols = zip(3*(sites-Nx*Ny), 3*(sites-Nx*Ny)+2, 3*(sites-Nx*(Ny-1)), \
                   3*(sites-Nx*(Ny-1))+2, 3*(sites-Nx), 3*(sites-Nx)+2, \
                   3*(sites-1), 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,\
                   3*(sites+1), 3*(sites+1)+2, 3*(sites+Nx*Ny), 3*(sites+Nx*Ny)+2)

    dfn_data = zip(defn_smNN, dv_smNN, defn_spN, dv_spN, defn_smN, dv_smN,\
                   defn_sm1, dv_sm1,  defn_s, defp_s, dv_s, defn_sp1, dv_sp1,\
                   defn_spNN, dv_spNN)

    update(dfn_rows, dfn_cols, dfn_data)

    #------------------------ fp derivatives ----------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    fp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = np.reshape(np.repeat(3*sites+1, 15), (len(sites), 15)).tolist()

    dfp_cols = zip(3*(sites-Nx*Ny)+1, 3*(sites-Nx*Ny)+2, 3*(sites-Nx*(Ny-1))+1,\
                   3*(sites-Nx*(Ny-1))+2, 3*(sites-Nx)+1, 3*(sites-Nx)+2, \
                   3*(sites-1)+1, 3*(sites-1)+2, 3*sites, 3*sites+1,\
                   3*sites+2, 3*(sites+1)+1, 3*(sites+1)+2, 3*(sites+Nx*Ny)+1,\
                   3*(sites+Nx*Ny)+2)

    dfp_data = zip(defp_smNN, dv_smNN, defp_spN, dv_spN, defp_smN, dv_smN,\
                   defp_sm1, dv_sm1, defn_s, defp_s, dv_s, defp_sp1, dv_sp1,\
                   defp_spNN, dv_spNN)

    update(dfp_rows, dfp_cols, dfp_data)

    #---------------- fv derivatives inside the system ------------------------
    # compute the derivatives
    dvmNN, dvmN, dvm1, defn, defp, dv, dvp1, dvpN, dvpNN = \
    fv_derivatives(sys, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = np.reshape(np.repeat(3*sites+2, 9), (len(sites), 9)).tolist()

    dfv_cols = zip(3*(sites-Nx*Ny)+2, 3*(sites-Nx*(Ny-1))+2, 3*(sites-Nx)+2, \
                   3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2, 3*(sites+1)+2,\
                   3*(sites+Nx*Ny)+2)

    dfv_data = zip(dvmNN, dvpN, dvmN, dvm1, defn, defp, dv, dvp1, dvpNN)

    update(dfv_rows, dfv_cols, dfv_data)



    ###########################################################################
    #           edges between contacts: 0 < i < Nx-1, j and k fixed           #
    ###########################################################################
    # Here we focus on the edges between the contacts. There are 4 cases again
    # (obviously).

    # lattice distances
    dx = sys.dx[1:]
    dxm1 = sys.dx[:-1]

    ###########################################################################
    #         edge z top // y back: 0 < i < Nx-1, j = Ny-1, k = Nz-1          #
    ###########################################################################
    # list of the sites
    sites = [i + (Ny-1)*Nx + (Nz-1)*Nx*Ny for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = np.repeat((sys.dy[0] + sys.dy[-1])/2., Nx-2)
    dz = np.repeat((sys.dz[0] + sys.dz[-1])/2., Nx-2)
    dym1 = np.repeat(sys.dy[-1], Nx-2)
    dzm1 = np.repeat(sys.dz[-1], Nx-2)

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites - Nx*(Ny-1)]
    s_spNN = [sites, sites - Nx*Ny*(Nz-1)]

    #------------------------ fn derivatives ----------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    fn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = np.reshape(np.repeat(3*sites, 15), (len(sites), 15)).tolist()

    dfn_cols = zip(3*(sites-Nx*Ny*(Nz-1)), 3*(sites-Nx*Ny*(Nz-1))+2, 3*(sites-Nx*Ny),\
                   3*(sites-Nx*Ny)+2, 3*(sites-Nx*(Ny-1)), 3*(sites-Nx*(Ny-1))+2,\
                   3*(sites-Nx), 3*(sites-Nx)+2, 3*(sites-1), 3*(sites-1)+2, \
                   3*sites, 3*sites+1, 3*sites+2, 3*(sites+1), 3*(sites+1)+2)

    dfn_data = zip(defn_spNN, dv_spNN, defn_smNN, dv_smNN, defn_spN, dv_spN,\
                   defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defn_sp1, dv_sp1)

    update(dfn_rows, dfn_cols, dfn_data)

    #------------------------ fp derivatives ----------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    fp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = np.reshape(np.repeat(3*sites+1, 15), (len(sites), 15)).tolist()

    dfp_cols = zip(3*(sites-Nx*Ny*(Nz-1))+1, 3*(sites-Nx*Ny*(Nz-1))+2, 3*(sites-Nx*Ny)+1,\
                   3*(sites-Nx*Ny)+2, 3*(sites-Nx*(Ny-1))+1, 3*(sites-Nx*(Ny-1))+2,\
                   3*(sites-Nx)+1, 3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2,\
                   3*sites, 3*sites+1, 3*sites+2, 3*(sites+1)+1, 3*(sites+1)+2)

    dfp_data = zip(defp_spNN, dv_spNN, defp_smNN, dv_smNN, defp_spN, dv_spN,\
                   defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defp_sp1, dv_sp1)

    update(dfp_rows, dfp_cols, dfp_data)

    #---------------- fv derivatives inside the system ------------------------
    # compute the derivatives
    dvmNN, dvmN, dvm1, defn, defp, dv, dvp1, dvpN, dvpNN = \
    fv_derivatives(sys, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = np.reshape(np.repeat(3*sites+2, 9), (len(sites), 9)).tolist()

    dfv_cols = zip(3*(sites-Nx*Ny*(Nz-1))+2, 3*(sites-Nx*Ny)+2, 3*(sites-Nx*(Ny-1))+2,\
                   3*(sites-Nx)+2, 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2, 3*(sites+1)+2)

    dfv_data = zip(dvpNN, dvmNN, dvpN, dvmN, dvm1, defn, defp, dv, dvp1)

    update(dfv_rows, dfv_cols, dfv_data)


    ###########################################################################
    #           edge z top // y front: 0 < i < Nx-1, j = 0, k = Nz-1          #
    ###########################################################################
    # list of the sites
    sites = [i + (Nz-1)*Nx*Ny for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = np.repeat(sys.dy[0], Nx-2)
    dz = np.repeat((sys.dz[0] + sys.dz[-1])/2., Nx-2)
    dym1 = np.repeat((sys.dy[0] + sys.dy[-1])/2., Nx-2)
    dzm1 = np.repeat(sys.dz[-1], Nx-2)

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites + Nx*(Ny-1), sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites - Nx*Ny*(Nz-1)]

    #------------------------ fn derivatives ----------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    fn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = np.reshape(np.repeat(3*sites, 15), (len(sites), 15)).tolist()

    dfn_cols = zip(3*(sites-Nx*Ny*(Nz-1)), 3*(sites-Nx*Ny*(Nz-1))+2, 3*(sites-Nx*Ny),\
                   3*(sites-Nx*Ny)+2, 3*(sites-1), 3*(sites-1)+2, 3*sites, \
                   3*sites+1, 3*sites+2, 3*(sites+1), 3*(sites+1)+2,\
                   3*(sites+Nx), 3*(sites+Nx)+2, 3*(sites+Nx*(Ny-1)), 3*(sites+Nx*(Ny-1))+2)

    dfn_data = zip(defn_spNN, dv_spNN, defn_smNN, dv_smNN, defn_sm1, dv_sm1,\
                   defn_s, defp_s, dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN,\
                   defn_smN, dv_smN)

    update(dfn_rows, dfn_cols, dfn_data)

    #------------------------ fp derivatives ----------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    fp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = np.reshape(np.repeat(3*sites+1, 15), (len(sites), 15)).tolist()

    dfp_cols = zip(3*(sites-Nx*Ny*(Nz-1))+1, 3*(sites-Nx*Ny*(Nz-1))+2, 3*(sites-Nx*Ny)+1,\
                   3*(sites-Nx*Ny)+2, 3*(sites-1)+1, 3*(sites-1)+2, 3*sites, \
                   3*sites+1, 3*sites+2, 3*(sites+1)+1, 3*(sites+1)+2, \
                   3*(sites+Nx)+1, 3*(sites+Nx)+2, 3*(sites+Nx*(Ny-1))+1, 3*(sites+Nx*(Ny-1))+2)

    dfp_data = zip(defp_spNN, dv_spNN, defp_smNN, dv_smNN, defp_sm1, dv_sm1,\
                   defn_s, defp_s, dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN,\
                   defp_smN, dv_smN)

    update(dfp_rows, dfp_cols, dfp_data)

    #---------------- fv derivatives inside the system ------------------------
    # compute the derivatives
    dvmNN, dvmN, dvm1, defn, defp, dv, dvp1, dvpN, dvpNN = \
    fv_derivatives(sys, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = np.reshape(np.repeat(3*sites+2, 9), (len(sites), 9)).tolist()

    dfv_cols = zip(3*(s-Nx*Ny*(Nz-1))+2, 3*(s-Nx*Ny)+2, 3*(s-1)+2, 3*s, 3*s+1,\
                   3*s+2, 3*(s+1)+2, 3*(s+Nx)+2, 3*(s+Nx*(Ny-1))+2)

    dfv_data = zip(dvpNN, dvmNN, dvm1, defn, defp, dv, dvp1, dvpN, dvmN)

    update(dfv_rows, dfv_cols, dfv_data)

    ###########################################################################
    #          edge z bottom // y back: 0 < i < Nx-1, j = Ny-1, k = 0         #
    ###########################################################################
    # list of the sites
    sites = [i + (Ny-1)*Nx for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = np.repeat((sys.dy[0] + sys.dy[-1])/2., Nx-2)
    dz = np.repeat(sys.dz[0], Nx-2)
    dym1 = np.repeat(sys.dy[-1], Nx-2)
    dzm1 = np.repeat((sys.dz[0] + sys.dz[-1])/2., Nx-2)

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites + Nx*Ny*(Nz-1), sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites - Nx*(Ny-1)]
    s_spNN = [sites, sites + Nx*Ny]

    #------------------------ fn derivatives ----------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    fn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = np.reshape(np.repeat(3*sites, 15), (len(sites), 15)).tolist()

    dfn_cols = zip(3*(sites-Nx*(Ny-1)), 3*(sites-Nx*(Ny-1))+2, 3*(sites-Nx), 3*(sites-Nx)+2,\
                   3*(sites-1), 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2, \
                   3*(sites+1), 3*(sites+1)+2, 3*(sites+Nx*Ny),\
                   3*(sites+Nx*Ny)+2, 3*(sites+Nx*Ny*(Nz-1)), 3*(sites+Nx*Ny*(Nz-1))+2)

    dfn_data = zip(defn_spN, dv_spN, defn_smN, dv_smN, defn_sm1, dv_sm1,\
                   defn_s, defp_s, dv_s, defn_sp1, dv_sp1, defn_spNN, dv_spNN,\
                   defn_smNN, dv_smNN)

    update(dfn_rows, dfn_cols, dfn_data)

    #------------------------ fp derivatives ----------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    fp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = np.reshape(np.repeat(3*sites+1, 15), (len(sites), 15)).tolist()

    dfp_cols = zip(3*(sites-Nx*(Ny-1))+1, 3*(sites-Nx*(Ny-1))+2, 3*(sites-Nx)+1,\
                   3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2, 3*sites, 3*sites+1,\
                   3*sites+2, 3*(sites+1)+1, 3*(sites+1)+2,\
                   3*(sites+Nx*Ny)+1, 3*(sites+Nx*Ny)+2, 3*(sites+Nx*Ny*(Nz-1))+1,\
                   3*(sites+Nx*Ny*(Nz-1))+2)

    dfp_data = zip(defp_spN, dv_spN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s,\
    defp_s, dv_s, defp_sp1, dv_sp1, defp_spNN, dv_spNN, defp_smNN, dv_smNN)

    update(dfp_rows, dfp_cols, dfp_data)

    #---------------- fv derivatives inside the system ------------------------
    # compute the derivatives
    dvmNN, dvmN, dvm1, defn, defp, dv, dvp1, dvpN, dvpNN = \
    fv_derivatives(sys, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = np.reshape(np.repeat(3*sites+2, 9), (len(sites), 9)).tolist()

    dfv_cols = zip(3*(sites-Nx*(Ny-1))+2, 3*(sites-Nx)+2, 3*(sites-1)+2, \
                   3*sites, 3*sites+1, 3*sites+2, 3*(sites+1)+2, \
                   3*(sites+Nx*Ny)+2, 3*(sites+Nx*Ny*(Nz-1))+2)

    dfv_data = zip(dvpN, dvmN, dvm1, defn, defp, dv, dvp1, dvpNN, dvmNN)

    update(dfv_rows, dfv_cols, dfv_data)

    ###########################################################################
    #         edge z bottom // y front: 0 < i < Nx-1, j = 0, k = 0            #
    ###########################################################################
    # list of the sites
    sites = [i for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = np.repeat(sys.dy[0], Nx-2)
    dz = np.repeat(sys.dz[0], Nx-2)
    dym1 = np.repeat((sys.dy[0] + sys.dy[-1])/2., Nx-2)
    dzm1 = np.repeat((sys.dz[0] + sys.dz[-1])/2., Nx-2)

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites + Nx*Ny*(Nz-1), sites]
    smN_s = [sites + Nx*(Ny-1), sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    #------------------------ fn derivatives ----------------------------------
    defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN, defn_spNN, dv_spNN = \
    fn_derivatives(sys, efn, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = np.reshape(np.repeat(3*sites, 15), (len(sites), 15)).tolist()

    dfn_cols = zip(3*(sites-1), 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,\
                   3*(sites+1), 3*(sites+1)+2, 3*(sites+Nx), 3*(sites+Nx)+2,\
                   3*(sites+Nx*(Ny-1)), 3*(sites+Nx*(Ny-1))+2, 3*(sites+Nx*Ny),\
                   3*(sites+Nx*Ny)+2, 3*(sites+Nx*Ny*(Nz-1)), 3*(sites+Nx*Ny*(Nz-1))+2)

    dfn_data = zip(defn_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defn_sp1, dv_sp1, defn_spN, dv_spN, defn_smN, dv_smN,\
                   defn_spNN, dv_spNN, defn_smNN, dv_smNN)

    update(dfn_rows, dfn_cols, dfn_data)

    #------------------------ fp derivatives ----------------------------------
    defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
    dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN, defp_spNN, dv_spNN = \
    fp_derivatives(sys, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,
    dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = np.reshape(np.repeat(3*sites+1, 15), (len(sites), 15)).tolist()

    dfp_cols = zip(3*(sites-1)+1, 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,\
                   3*(sites+1)+1, 3*(sites+1)+2, 3*(sites+Nx)+1, 3*(sites+Nx)+2,\
                   3*(sites+Nx*(Ny-1))+1, 3*(sites+Nx*(Ny-1))+2,\
                   3*(sites+Nx*Ny)+1, 3*(sites+Nx*Ny)+2, 3*(sites+Nx*Ny*(Nz-1))+1,\
                   3*(sites+Nx*Ny*(Nz-1))+2)

    dfp_data = zip(defp_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defp_sp1, dv_sp1, defp_spN, dv_spN, defp_smN, dv_smN,\
                   defp_spNN, dv_spNN, defp_smNN, dv_smNN)

    update(dfp_rows, dfp_cols, dfp_data)

    #---------------- fv derivatives inside the system ------------------------
    # compute the derivatives
    dvmNN, dvmN, dvm1, defn, defp, dv, dvp1, dvpN, dvpNN = \
    fv_derivatives(sys, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = np.reshape(np.repeat(3*sites+2, 9), (len(sites), 9)).tolist()

    dfv_cols = zip(3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2, 3*(sites+1)+2,\
                   3*(sites+Nx)+2, 3*(sites+Nx*(Ny-1))+2, 3*(sites+Nx*Ny)+2,\
                   3*(sites+Nx*Ny*(Nz-1))+2)

    dfv_data = zip(dvm1, defn, defp, dv, dvp1, dvpN, dvmN, dvpNN, dvmNN)

    update(dfv_rows, dfv_cols, dfv_data)


    J = coo_matrix((data, (rows, columns)), shape=(3*Nx*Ny*Nz, 3*Nx*Ny*Nz), dtype=np.float64)
    return J
