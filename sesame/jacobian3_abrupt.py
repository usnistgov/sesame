# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import numpy as np
from itertools import chain

from .observables import *
from .defects  import defectsJ

def getJ(sys, v, efn, efp):
    ###########################################################################
    #                     organization of the Jacobian matrix                 #
    ###########################################################################
    # A site with coordinates (i,j,k) corresponds to a site number s as follows:
    # k = s//(Nx*Ny)
    # j = s - s//Nx
    # i = s - j*Nx - k*Nx*Ny
    # 
    # Note that the boundary conditions on the contacts out of equilibrium are
    # Dirichlet, whatever the equilibrium boundary conditions were (so the
    # equilibrium values are conserved).

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
    _sites = np.arange(Nx*Ny*Nz, dtype=int)

    # carrier densities
    n = sys.Nc * np.exp(+sys.bl + efn + v)
    p = sys.Nv * np.exp(-sys.Eg - sys.bl + efp - v)

    # bulk charges
    drho_defn_s = - n
    drho_defp_s = + p
    drho_dv_s = - n - p

    # derivatives of the bulk recombination rates
    dr_defn_s, dr_defp_s, dr_dv_s = get_bulk_rr_derivs(sys, n, p)

    # charge defects
    if len(sys.defects_list) != 0:
        defectsJ(sys, n, p, drho_dv_s, drho_defn_s, drho_defp_s, dr_defn_s,\
                 dr_defp_s, dr_dv_s)

    # charge is divided by epsilon
    drho_defn_s = drho_defn_s / sys.epsilon[_sites]
    drho_defp_s = drho_defp_s / sys.epsilon[_sites]
    drho_dv_s = drho_dv_s / sys.epsilon[_sites]

    # reshape the array as array[y-indices, x-indices]
    _sites = np.arange(Nx*Ny*Nz, dtype=int).reshape(Nz, Ny, Nx)

    def update(r, c, d):
        global rows, columns, data
        rows += list(chain.from_iterable(r))
        columns += list(chain.from_iterable(c))
        data += list(chain.from_iterable(d))

    def current_derivs(get_j_derivs, sys, ef, v, dx, dxm1, dy, dym1, dz, dzm1, sites):
        djx_s_def_s, djx_s_def_sp1, djx_s_dv_s, djx_s_dv_sp1 = 0, 0, 0, 0
        djx_sm1_def_sm1, djx_sm1_def_s, djx_sm1_dv_sm1, djx_sm1_dv_s = \
        0, 0, 0, 0
        djy_s_def_s, djy_s_def_spN, djy_s_dv_s, djy_s_dv_spN = 0, 0, 0, 0
        djy_smN_def_smN, djy_smN_def_s, djy_smN_dv_smN, djy_smN_dv_s = \
        0, 0, 0, 0
        djz_s_def_s, djz_s_def_spNN, djz_s_dv_s, djz_s_dv_spNN = \
        0, 0, 0, 0
        djz_smNN_def_smNN, djz_smNN_def_s, djz_smNN_dv_smNN,\
        djz_smNN_dv_s = 0, 0, 0, 0

        if dx.all() != 0:
            djx_s_def_s, djx_s_def_sp1, djx_s_dv_s, djx_s_dv_sp1 = \
            get_j_derivs(sys, ef, v, sites, sites+1, dx)

        if dxm1.all() != 0:
            djx_sm1_def_sm1, djx_sm1_def_s, djx_sm1_dv_sm1,\
            djx_sm1_dv_s = get_j_derivs(sys, ef, v, sites-1, sites, dxm1)

        if dy.all() != 0:
            djy_s_def_s, djy_s_def_spN, djy_s_dv_s, djy_s_dv_spN = \
            get_j_derivs(sys, ef, v, sites, sites+Nx, dy)

        if dym1.all() != 0:
            djy_smN_def_smN, djy_smN_def_s, djy_smN_dv_smN,\
            djy_smN_dv_s = get_j_derivs(sys, ef, v, sites-Nx, sites, dym1)

        if dz.all() != 0:
            djz_s_def_s, djz_s_def_spNN, djz_s_dv_s, djz_s_dv_spNN = \
            get_j_derivs(sys, ef, v, sites, sites+Nx*Ny, dz)

        if dzm1.all() != 0:
            djz_smNN_def_smNN, djz_smNN_def_s, djz_smNN_dv_smNN,\
            djz_smNN_dv_s = get_j_derivs(sys, ef, v, sites-Nx*Ny, sites, dzm1)

        return djx_s_def_s, djx_s_def_sp1, djx_s_dv_s, djx_s_dv_sp1,\
        djx_sm1_def_sm1, djx_sm1_def_s, djx_sm1_dv_sm1, djx_sm1_dv_s,\
        djy_s_def_s, djy_s_def_spN, djy_s_dv_s, djy_s_dv_spN,\
        djy_smN_def_smN, djy_smN_def_s, djy_smN_dv_smN, djy_smN_dv_s,\
        djz_s_def_s, djz_s_def_spNN, djz_s_dv_s, djz_s_dv_spNN,\
        djz_smNN_def_smNN, djz_smNN_def_s, djz_smNN_dv_smNN, djz_smNN_dv_s

    def dd_derivs(carriers, sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites):
    # Derivatives of the right hand side fn, fp, fv

        # lattice distances
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        # derivatives of the currents
        if carriers == 'holes':
            get_j_derivs = get_jp_derivs
            ef = efp
        if carriers == 'electrons':
            get_j_derivs = get_jn_derivs
            ef = efn

        djx_s_def_s, djx_s_def_sp1, djx_s_dv_s, djx_s_dv_sp1,\
        djx_sm1_def_sm1, djx_sm1_def_s, djx_sm1_dv_sm1, djx_sm1_dv_s,\
        djy_s_def_s, djy_s_def_spN, djy_s_dv_s, djy_s_dv_spN,\
        djy_smN_def_smN, djy_smN_def_s, djy_smN_dv_smN, djy_smN_dv_s,\
        djz_s_def_s, djz_s_def_spNN, djz_s_dv_s, djz_s_dv_spNN,\
        djz_smNN_def_smNN, djz_smNN_def_s, djz_smNN_dv_smNN, djz_smNN_dv_s\
        = current_derivs(get_j_derivs, sys, ef, v, dx, dxm1, dy, dym1, dz, dzm1, sites) 

        # compute derivatives of fn or fp
        def_smNN = - djz_smNN_def_smNN / dzbar
        dv_smNN = - djz_smNN_dv_smNN / dzbar

        def_smN = - djy_smN_def_smN / dybar
        dv_smN = - djy_smN_dv_smN / dybar

        def_sm1 = - djx_sm1_def_sm1 / dxbar
        dv_sm1 = - djx_sm1_dv_sm1 / dxbar

        dv_s = (djx_s_dv_s - djx_sm1_dv_s) / dxbar +\
               (djy_s_dv_s - djy_smN_dv_s) / dybar +\
               (djz_s_dv_s - djz_smNN_dv_s) / dzbar 
        def_s = (djx_s_def_s - djx_sm1_def_s) / dxbar +\
                (djy_s_def_s - djy_smN_def_s) / dybar +\
                (djz_s_def_s - djz_smNN_def_s) / dzbar 
        if carriers == 'holes':
            defn_s =  dr_defn_s[sites]
            defp_s = def_s + dr_defp_s[sites]
            dv_s = dv_s + dr_dv_s[sites]
        if carriers == 'electrons':
            defn_s =  def_s - dr_defn_s[sites]
            defp_s = - dr_defp_s[sites]
            dv_s = dv_s - dr_dv_s[sites]

        def_sp1 = djx_s_def_sp1 / dxbar
        dv_sp1 = djx_s_dv_sp1 / dxbar

        def_spN = djy_s_def_spN / dybar
        dv_spN = djy_s_dv_spN / dybar

        def_spNN = djz_s_def_spNN / dzbar
        dv_spNN = djz_s_dv_spNN / dzbar

        return def_smNN, dv_smNN, def_smN, dv_smN, def_sm1, dv_sm1, \
               defn_s, defp_s, dv_s, def_sp1, dv_sp1, def_spN, dv_spN,\
               def_spNN, dv_spNN

    def ddp_derivs(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites):
        # fn derivatives -------------------------------------
        defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, \
        defn_s, defp_s, dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN,\
        defn_spNN, dv_spNN =\
        dd_derivs('electrons', sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

        dfn_rows = np.reshape(np.repeat(3*sites, 15), (len(sites), 15)).tolist()
 
        dfn_cols = zip(3*(sites-Nx*Ny), 3*(sites-Nx*Ny)+2, 3*(sites-Nx),\
                       3*(sites-Nx)+2, 3*(sites-1), 3*(sites-1)+2, 3*sites,\
                       3*sites+1, 3*sites+2, 3*(sites+1), 3*(sites+1)+2, 3*(sites+Nx),\
                       3*(sites+Nx)+2, 3*(sites+Nx*Ny), 3*(sites+Nx*Ny)+2)
 
        dfn_data = zip(defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, \
                        defn_s, defp_s, dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN,\
                        defn_spNN, dv_spNN)
 
        update(dfn_rows, dfn_cols, dfn_data)

        # fp derivatives -------------------------------------
        defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, \
        defn_s, defp_s, dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN,\
        defp_spNN, dv_spNN =\
        dd_derivs('holes', sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

        dfp_rows = np.reshape(np.repeat(3*sites+1, 15), (len(sites), 15)).tolist()
 
        dfp_cols = zip(3*(sites-Nx*Ny)+1, 3*(sites-Nx*Ny)+2, 3*(sites-Nx)+1,\
                       3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2, 3*sites,\
                       3*sites+1, 3*sites+2, 3*(sites+1)+1, 3*(sites+1)+2,\
                       3*(sites+Nx)+1, 3*(sites+Nx)+2, 3*(sites+Nx*Ny)+1, 3*(sites+Nx*Ny)+2)
 
        dfp_data = zip(defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, \
                        defn_s, defp_s, dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN,\
                        defp_spNN, dv_spNN)
 
        update(dfp_rows, dfp_cols, dfp_data)

        # fv derivatives -------------------------------------
        # lattice distances
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        # compute differences of potentials
        dv_sm1, dv_sp1, dv_smN, dv_spN, dv_smNN, dv_spNN = 0, 0, 0, 0, 0, 0
        dv = -drho_dv_s[sites]
        defn = - drho_defn_s[sites]
        defp = - drho_defp_s[sites]

        global rows, columns, data
        if dx.all() != 0:
            dv_sp1 = -1 / (dx * dxbar)
            dv += -dv_sp1
            rows += (3*sites+2).tolist()
            columns += (3*(sites+1)+2).tolist()
            data += dv_sp1.tolist()
        if dxm1.all() != 0:
            dv_sm1 = -1 / (dxm1 * dxbar)
            dv += -dv_sm1
            rows += (3*sites+2).tolist()
            columns += (3*(sites-1)+2).tolist()
            data += dv_sm1.tolist()
        if dy.all() != 0:
            dv_spN = -1 / (dy * dybar)
            dv += -dv_spN
            rows += (3*sites+2).tolist()
            columns += (3*(sites+Nx)+2).tolist()
            data += dv_spN.tolist()
        if dym1.all() != 0:
            dv_smN = 1 / (dym1 * dybar)
            dv += -dv_smN
            rows += (3*sites+2).tolist()
            columns += (3*(sites-Nx)+2).tolist()
            data += dv_smN.tolist()
        if dz.all() != 0:
            dv_spNN = -1 / (dz * dzbar)
            dv += -dv_spNN
            rows += (3*sites+2).tolist()
            columns +=(3*(sites+Nx*Ny)+2).tolist()
            data += dv_spNN.tolist()
        if dzm1.all() != 0:
            dv_smNN = -1 / (dzm1 * dzbar)
            dv += -dv_smNN
            rows += (3*sites+2).tolist()
            columns += (3*(sites-Nx*Ny)+2).tolist()
            data += dv_smNN.tolist()

        rows += (3*sites+2).tolist()
        columns += (3*sites+2).tolist()
        data += dv.tolist()

        rows += (3*sites+2).tolist()
        columns += (3*sites).tolist()
        data += defn.tolist()

        rows += (3*sites+2).tolist()
        columns += (3*sites+1).tolist()
        data += defp.tolist()


    def bnp_derivs(carriers, sys, ef, v, dy, dym1, dz, dzm1, sites):
    # Derivatives of the right hand side bn, bp

        # lattice distances
        dx = np.array([0])
        dxm1 = sys.dx[-1]
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        # derivatives of the currents
        if carriers == 'holes':
            get_j_derivs = get_jp_derivs
        if carriers == 'electrons':
            get_j_derivs = get_jn_derivs

        # compute the derivatives of the currents
        _, _, _, _,\
        djx_sm1_def_sm1, djx_sm1_def_s, djx_sm1_dv_sm1, djx_sm1_dv_s,\
        djy_s_def_s, djy_s_def_spN, djy_s_dv_s, djy_s_dv_spN,\
        djy_smN_def_smN, djy_smN_def_s, djy_smN_dv_smN, djy_smN_dv_s,\
        djz_s_def_s, djz_s_def_spNN, djz_s_dv_s, djz_s_dv_spNN,\
        djz_smNN_def_smNN, djz_smNN_def_s, djz_smNN_dv_smNN, djz_smNN_dv_s\
        = current_derivs(get_j_derivs, sys, ef, v, dx, dxm1, dy, dym1, dz, dzm1, sites) 

        # compute bn derivatives
        def_smNN = dxbar/dzbar * djz_smNN_def_smNN
        dv_smNN = dxbar/dzbar * djz_smNN_dv_smNN

        def_smN = dxbar/dybar * djy_smN_def_smN
        dv_smN = dxbar/dybar * djy_smN_dv_smN
        
        def_sm1 = djx_sm1_def_sm1
        dv_sm1 = djx_sm1_dv_sm1

        def_s = djx_sm1_def_s + dxbar * (\
              - (djy_s_def_s - djy_smN_def_s) / dybar\
              - (djy_s_def_s - djz_smNN_def_s) / dzbar)
        dv_s = djx_sm1_dv_s + dxbar * (dr_dv_s[sites]\
             - (djy_s_dv_s  - djy_smN_dv_s) / dybar\
             - (djz_s_dv_s  - djz_smNN_dv_s) / dzbar)
        if carriers == 'electrons':
            defn_s = def_s +  dxbar * dr_defn_s[sites] + sys.Scn[1] * n[sites]
            defp_s = dxbar * dr_defp_s[sites]
            dv_s = dv_s + dxbar * dr_dv_s[sites] + sys.Scn[1] * n[sites]

        if carriers == 'holes':
            defn_s = - dxbar * dr_defn_s[sites]
            defp_s = def_s - dxbar * dr_defp_s[sites] - sys.Scp[1] * p[sites]
            dv_s = dv_s - dxbar * dr_dv_s[sites] + sys.Scp[1] * p[sites]

        def_spN = - dxbar/dybar * djy_s_def_spN
        dv_spN = - dxbar/dybar * djy_s_dv_spN

        def_spNN = - dxbar/dzbar * djz_s_def_spNN
        dv_spNN = - dxbar/dzbar * djz_s_dv_spNN
        
        return def_smNN, dv_smNN, def_smN, dv_smN, def_sm1, dv_sm1, defn_s,\
               defp_s, dv_s, def_spN, dv_spN, def_spNN, dv_spNN

    def right_bc_derivs(sys, efn, efp, v, dy, dym1, dz, dzm1, sites):
        # bn derivatives -------------------------------------
        defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
        dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN = \
        bnp_derivs('electrons', sys, efn, v, dy, dym1, dz, dzm1, sites)

        dbn_rows = np.reshape(np.repeat(3*sites, 13), (len(sites), 13)).tolist()

        dbn_cols = zip(3*(sites-Nx*Ny), 3*(sites-Nx*Ny)+2, 3*(sites-Nx),
                       3*(sites-Nx)+2, 3*(sites-1), 3*(sites-1)+2, 3*sites,\
                       3*sites+1, 3*sites+2, 3*(sites+Nx), 3*(sites+Nx)+2,\
                       3*(sites+Nx*Ny), 3*(sites+Nx*Ny)+2)

        dbn_data = zip(defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1,\
                       defn_s, defp_s, dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN)

        update(dbn_rows, dbn_cols, dbn_data)

        # bp derivatives -------------------------------------
        defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
        dv_s, defp_spN, dv_spN, defp_spNN, dv_spNN = \
        bnp_derivs('holes', sys, efp, v, dy, dym1, dz, dzm1, sites)

        dbp_rows = np.reshape(np.repeat(3*sites+1, 13), (len(sites), 13)).tolist()

        dbp_cols = zip(3*(sites-Nx*Ny)+1, 3*(sites-Nx*Ny)+2, 3*(sites-Nx)+1,
                       3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2, 3*sites,\
                       3*sites+1, 3*sites+2, 3*(sites+Nx)+1, 3*(sites+Nx)+2,\
                       3*(sites+Nx*Ny)+1, 3*(sites+Nx*Ny)+2)

        dbp_data = zip(defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1,\
                       defn_s, defp_s, dv_s, defp_spN, dv_spN, defp_spNN, dv_spNN)

        update(dbp_rows, dbp_cols, dbp_data)

        # bv derivatives -------------------------------------
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
    sites = _sites[1:Nz-1, 1:Ny-1, 1:Nx-1].flatten()

    # lattice distances
    dx = np.tile(sys.dx[1:], (Ny-2)*(Nz-2))
    dy = np.repeat(sys.dy[1:], (Nx-2)*(Nz-2)) 
    dz = np.repeat(sys.dz[1:], (Nx-2)*(Ny-2))
    dxm1 = np.tile(sys.dx[:-1], (Ny-2)*(Nz-2))
    dym1 = np.repeat(sys.dy[:-1], (Nx-2)*(Nz-2))
    dzm1 = np.repeat(sys.dz[:-1], (Nx-2)*(Ny-2))

    ddp_derivs(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #        left boundary: i = 0, 0 <= j <= Ny-1, 0 <= k <= Nz-1             #
    ###########################################################################
    # We compute an, ap, av derivatives. Those functions are only defined on the
    # left boundary of the system.

    # list of the sites on the left side
    sites = _sites[:, :, 0].flatten()

    #-------------------------- an derivatives --------------------------------
    defn_s, defn_sp1, dv_s, dv_sp1 = get_jn_derivs(sys, efn, v, sites, sites+1, sys.dx[0])

    defn_s -= sys.Scn[0] * n[sites]
    dv_s -= sys.Scn[0] * n[sites]

    # update the sparse matrix row and columns
    dan_rows = np.reshape(np.repeat(3*sites, 4), (len(sites), 4)).tolist()

    dan_cols = zip(3*sites, 3*sites+2, 3*(sites+1), 3*(sites+1)+2)

    dan_data = zip(defn_s, dv_s, defn_sp1, dv_sp1)

    update(dan_rows, dan_cols, dan_data)

    #-------------------------- ap derivatives --------------------------------
    defp_s, defp_sp1, dv_s, dv_sp1 = get_jp_derivs(sys, efp, v, sites, sites+1, sys.dx[0])

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
    dav_data = np.ones((len(sites,))).tolist()

    rows += dav_rows
    columns += dav_cols
    data += dav_data

    ###########################################################################
    #         right boundary: i = Nx-1, 0 < j < Ny-1, 0 < k < Nz-1            #
    ###########################################################################
    # We compute bn, bp, bv derivatives. Those functions are only defined on the
    # right boundary of the system.

    # list of the sites on the right side
    sites = _sites[1:Nz-1, 1:Ny-1, Nx-1].flatten()

    # dy and dz
    dy = np.repeat(sys.dy[1:], Nz-2)
    dym1 = np.repeat(sys.dy[:-1], Nz-2)
    dz = np.repeat(sys.dz[1:], Ny-2)
    dzm1 = np.repeat(sys.dz[:-1], Ny-2)

    right_bc_derivs(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #           right boundary: i = Nx-1, j = Ny-1, 0 < k < Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[1:Nz-1, Ny-1, Nx-1].flatten()

    # lattice distances
    dy = np.array([0])
    dym1 = np.repeat(sys.dy[-1], Nz-2) 
    dz = sys.dz[1:]
    dzm1 = sys.dz[:-1]

    right_bc_derivs(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #              right boundary: i = Nx-1, j = 0, 0 < k < Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[1:Nz-1, 0, Nx-1].flatten()

    # lattice distances
    dy = np.repeat(sys.dy[-1], Nz-2)
    dym1 =  np.array([0])
    dz = sys.dz[1:]
    dzm1 = sys.dz[:-1]

    right_bc_derivs(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #           right boundary: i = Nx-1, 0 < j < Ny-1, k = Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[Nz-1, 1:Ny-1, Nx-1].flatten()

    # lattice distances
    dy = sys.dy[1:]
    dym1 = sys.dy[:-1]
    dz = np.array([0])
    dzm1 = np.repeat(sys.dz[-1], Ny-2)

    right_bc_derivs(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #              right boundary: i = Nx-1, 0 < j < Ny-1, k = 0              #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[0, 1:Ny-1, Nx-1].flatten()

    # lattice distances
    dy = sys.dy[1:]
    dym1 = sys.dy[:-1]
    dz = np.repeat(sys.dz[0], Ny-2)
    dzm1 = np.array([0])

    right_bc_derivs(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = Ny-1, k = 0              #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[0, Ny-1, Nx-1].flatten()

    # lattice distances
    dy = np.array([0])
    dym1 = sys.dy[-1]
    dz = sys.dz[0]
    dzm1 = np.array([0])

    right_bc_derivs(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = Ny-1, k = Nz-1           #
    ###########################################################################
    # list of the sites on the right side
    sites = np.array([Nx-1 + (Ny-1)*Nx + (Nz-1)*Nx*Ny])

    # lattice distances
    dy = np.array([0])
    dym1 = sys.dy[-1]
    dz = np.array([0])
    dzm1 = sys.dz[-1]

    right_bc_derivs(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = 0, k = 0                 #
    ###########################################################################
    # list of the sites on the right side
    sites = np.array([Nx-1])

    # lattice distances
    dy = sys.dy[0]
    dym1 = np.array([0])
    dz = sys.dz[0]
    dzm1 = np.array([0])

    right_bc_derivs(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = 0, k = Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = np.array([Nx-1 + (Nz-1)*Nx*Ny])

    # lattice distances
    dy = sys.dy[0]
    dym1 = np.array([0])
    dz = np.array([0])
    dzm1 = sys.dz[0]

    right_bc_derivs(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)



    ###########################################################################
    #            faces between contacts: 0 < i < Nx-1, j or k fixed           #
    ###########################################################################
    # Here we focus on the faces between the contacts. There are 4 cases
    # (obviously).

    ###########################################################################
    #              z-face top: 0 < i < Nx-1, 0 < j < Ny-1, k = Nz-1           #
    ###########################################################################
    # list of the sites
    sites = _sites[Nz-1, 1:Ny-1, 1:Nx-1].flatten()

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny-2)
    dy = np.repeat(sys.dy[1:], Nx-2)
    dz = np.array([0])
    dxm1 = np.tile(sys.dx[:-1], Ny-2)
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dzm1 = np.repeat(sys.dz[-1], (Nx-2)*(Ny-2))

    ddp_derivs(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #             z- face bottom: 0 < i < Nx-1, 0 < j < Ny-1, k = 0           #
    ###########################################################################
    # list of the sites
    sites = _sites[0, 1:Ny-1, 1:Nx-1].flatten()

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny-2)
    dy = np.repeat(sys.dy[1:], Nx-2)
    dz = np.repeat(sys.dz[0], (Nx-2)*(Ny-2))
    dxm1 = np.tile(sys.dx[:-1], Ny-2)
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dzm1 = np.array([0])

    ddp_derivs(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #            y-face front: 0 < i < Nx-1, j = 0, 0 < k < Nz-1              #
    ###########################################################################
    # list of the sites
    sites = _sites[1:Nz-1, 0, 1:Nx-1].flatten()

    # lattice distances
    dx = np.tile(sys.dx[1:], Nz-2)
    dy = np.repeat(sys.dy[0], (Nx-2)*(Nz-2))
    dz = np.repeat(sys.dz[1:], (Nx-2))
    dxm1 = np.tile(sys.dx[:-1], Nz-2)
    dym1 = np.array([0])
    dzm1 = np.repeat(sys.dz[:-1], Nx-2)

    ddp_derivs(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #            y-face back: 0 < i < Nx-1, j = Ny-1, 0 < k < Nz-1            #
    ###########################################################################
    # list of the sites
    sites = _sites[1:Nz-1, Ny-1, 1:Nx-1].flatten()

    # lattice distances
    dx = np.tile(sys.dx[1:], Nz-2)
    dy = np.array([0])
    dz = np.repeat(sys.dz[1:], Nx-2)
    dxm1 = np.tile(sys.dx[:-1], Nz-2)
    dym1 = np.repeat(sys.dy[0], (Nx-2)*(Nz-2))
    dzm1 = np.repeat(sys.dz[:-1], Nx-2)

    ddp_derivs(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)


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
    sites = _sites[Nz-1, Ny-1, 1:Nx-1].flatten()

    # lattice distances
    dy = np.array([0])
    dz = np.array([0])
    dym1 = np.repeat(sys.dy[-1], Nx-2)
    dzm1 = np.repeat(sys.dz[-1], Nx-2)

    ddp_derivs(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #           edge z top // y front: 0 < i < Nx-1, j = 0, k = Nz-1          #
    ###########################################################################
    # list of the sites
    sites = _sites[Nz-1, 0, 1:Nx-1].flatten()

    # lattice distances
    dy = np.repeat(sys.dy[0], Nx-2)
    dz = np.array([0])
    dym1 = np.array([0])
    dzm1 = np.repeat(sys.dz[-1], Nx-2)

    ddp_derivs(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #          edge z bottom // y back: 0 < i < Nx-1, j = Ny-1, k = 0         #
    ###########################################################################
    # list of the sites
    sites = _sites[0, Ny-1, 1:Nx-1].flatten()

    # lattice distances
    dy = np.array([0])
    dz = np.repeat(sys.dz[0], Nx-2)
    dym1 = np.repeat(sys.dy[-1], Nx-2)
    dzm1 = np.array([0])

    ddp_derivs(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #         edge z bottom // y front: 0 < i < Nx-1, j = 0, k = 0            #
    ###########################################################################
    # list of the sites
    sites = _sites[0, 0, 1:Nx-1].flatten()

    # lattice distances
    dy = np.repeat(sys.dy[0], Nx-2)
    dz = np.repeat(sys.dz[0], Nx-2)
    dym1 = np.array([0])
    dzm1 = np.array([0])

    ddp_derivs(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)


    # remove data that are outside the system
    rows = [i for idx, i in enumerate(rows) if 0 <= columns[idx] < 3*Nx*Ny*Nz]
    data = [i for idx, i in enumerate(data) if 0 <= columns[idx] < 3*Nx*Ny*Nz]
    columns = [i for i in columns if 0 <= i < 3*Nx*Ny*Nz]

    return rows, columns, data
