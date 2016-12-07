import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from itertools import chain

from .observables import *

def getJ(sys, v, efn, efp, with_mumps):
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
    drho_defp_s = + p
    drho_dv_s = - n - p

    # derivatives of the bulk recombination rates
    dr_defn_s, dr_defp_s, dr_dv_s = \
    get_rr_derivs(sys, n, p, sys.n1, sys.p1, sys.tau_e, sys.tau_h, sites)\

    # extra charge density
    if hasattr(sys, 'Nextra'): 
        # find sites containing extra charges
        for idx, matches in enumerate(sys.extra_charge_sites):
            nextra = sys.nextra[idx, matches]
            pextra = sys.pextra[idx, matches]
            _n = n[matches]
            _p = p[matches]

            # extra charge density
            Se = sys.Seextra[idx, matches]
            Sh = sys.Shextra[idx, matches]
            d = (Se*(_n+nextra)+Sh*(_p+pextra))**2
            drho_defn_s[matches] += - sys.Nextra[idx, matches] *\
                Se*_n * (Se*nextra + Sh*_p) / d
            drho_defp_s[matches] += sys.Nextra[idx, matches] *\
                (Se*_n + Sh*pextra) * Sh*_p / d
            drho_dv_s[matches] += - sys.Nextra[idx, matches] *\
                (Se**2*_n*nextra + 2*Sh*Se*_p*_n + Sh**2*_p*pextra) / d

            # extra charge recombination
            defn, defp, dv =  get_rr_derivs(sys, _n, _p, nextra, pextra, 1/Se, 1/Sh, matches)
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

    def current_derivs(get_j_derivs, sys, ef, v, smNN_s, smN_s, s_spN, s_spNN,\
                       dx, dxm1, dy, dym1, dz, dzm1, sites):
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

        djx_sm1_def_sm1, djx_sm1_def_s, djx_sm1_dv_sm1,\
        djx_sm1_dv_s = get_j_derivs(sys, ef, v, sites-1, sites, dxm1)

        djy_s_def_s, djy_s_def_spN, djy_s_dv_s, djy_s_dv_spN = \
        get_j_derivs(sys, ef, v, s_spN[0], s_spN[1], dy)

        djy_smN_def_smN, djy_smN_def_s, djy_smN_dv_smN,\
        djy_smN_dv_s = get_j_derivs(sys, ef, v, smN_s[0], smN_s[1], dym1)

        djz_s_def_s, djz_s_def_spNN, djz_s_dv_s, djz_s_dv_spNN = \
        get_j_derivs(sys, ef, v, s_spNN[0], s_spNN[1], dz)

        djz_smNN_def_smNN, djz_smNN_def_s, djz_smNN_dv_smNN,\
        djz_smNN_dv_s = get_j_derivs(sys, ef, v, smNN_s[0], smNN_s[1], dzm1)

        return djx_s_def_s, djx_s_def_sp1, djx_s_dv_s, djx_s_dv_sp1,\
        djx_sm1_def_sm1, djx_sm1_def_s, djx_sm1_dv_sm1, djx_sm1_dv_s,\
        djy_s_def_s, djy_s_def_spN, djy_s_dv_s, djy_s_dv_spN,\
        djy_smN_def_smN, djy_smN_def_s, djy_smN_dv_smN, djy_smN_dv_s,\
        djz_s_def_s, djz_s_def_spNN, djz_s_dv_s, djz_s_dv_spNN,\
        djz_smNN_def_smNN, djz_smNN_def_s, djz_smNN_dv_smNN, djz_smNN_dv_s

    def dd_derivs(carriers, sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                  dx, dxm1, dy, dym1, dz, dzm1, sites):

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
        = current_derivs(get_j_derivs, sys, ef, v, smNN_s, smN_s, s_spN,\
                         s_spNN, dx, dxm1, dy, dym1, dz, dzm1, sites) 

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

    def ddp_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                   dx, dxm1, dy, dym1, dz, dzm1, sites):
        # fn derivatives ------------------------------------------
        defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, \
        defn_s, defp_s, dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN,\
        defn_spNN, dv_spNN =\
        dd_derivs('electrons', sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                  dx, dxm1, dy, dym1, dz, dzm1, sites)

        dfn_rows = np.reshape(np.repeat(3*sites, 15), (len(sites), 15)).tolist()
 
        dfn_cols = zip(3*smNN_s[0], 3*smNN_s[0]+2, 3*smN_s[0], 3*smN_s[0]+2,\
                       3*(sites-1), 3*(sites-1)+2, 3*sites, 3*sites+1,\
                       3*sites+2, 3*(sites+1), 3*(sites+1)+2, 3*s_spN[1],\
                       3*s_spN[1]+2, 3*s_spNN[1], 3*s_spNN[1]+2)
 
        dfn_data = zip(defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, \
                        defn_s, defp_s, dv_s, defn_sp1, dv_sp1, defn_spN, dv_spN,\
                        defn_spNN, dv_spNN)
 
        update(dfn_rows, dfn_cols, dfn_data)

        # fp derivatives ------------------------------------------
        defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, \
        defn_s, defp_s, dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN,\
        defp_spNN, dv_spNN =\
        dd_derivs('holes', sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                  dx, dxm1, dy, dym1, dz, dzm1, sites)

        dfp_rows = np.reshape(np.repeat(3*sites+1, 15), (len(sites), 15)).tolist()
 
        dfp_cols = zip(3*smNN_s[0]+1, 3*smNN_s[0]+2, 3*smN_s[0]+1,\
                       3*smN_s[0]+2, 3*(sites-1)+1, 3*(sites-1)+2, 3*sites,\
                       3*sites+1, 3*sites+2, 3*(sites+1)+1, 3*(sites+1)+2,\
                       3*s_spN[1]+1, 3*s_spN[1]+2, 3*s_spNN[1]+1, 3*s_spNN[1]+2)
 
        dfp_data = zip(defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, \
                        defn_s, defp_s, dv_s, defp_sp1, dv_sp1, defp_spN, dv_spN,\
                        defp_spNN, dv_spNN)
 
        update(dfp_rows, dfp_cols, dfp_data)

        # fv derivatives ------------------------------------------
        # lattice distances
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        # compute the derivatives
        dvmNN = -1./(dzm1 * dzbar)
        dvmN = -1./(dym1 * dybar)
        dvm1 = -1./(dxm1 * dxbar)
        dv = 2./(dx * dxm1) + 2./(dy * dym1) + 2./(dz * dzm1) - drho_dv_s[sites]
        defn = - drho_defn_s[sites]
        defp = - drho_defp_s[sites]
        dvp1 = -1./(dx * dxbar)
        dvpN = -1./(dy * dybar)
        dvpNN = -1./(dz * dzbar)
        
        dfv_rows = np.reshape(np.repeat(3*sites+2, 9), (len(sites), 9)).tolist()

        dfv_cols = zip(3*smNN_s[0]+2, 3*smN_s[0]+2, 3*(sites-1)+2, 3*sites,\
                       3*sites+1, 3*sites+2, 3*(sites+1)+2, 3*s_spN[1]+2,\
                       3*s_spNN[1]+2)

        dfv_data = zip(dvmNN, dvmN, dvm1, defn, defp, dv, dvp1, dvpN, dvpNN)

        update(dfv_rows, dfv_cols, dfv_data)


    def bnp_derivs(carriers, sys, ef, v, smNN_s, smN_s, s_spN, s_spNN,\
                   dy, dym1, dz, dzm1, sites):
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
        = current_derivs(get_j_derivs, sys, ef, v, smNN_s, smN_s, s_spN,\
                         s_spNN, dx, dxm1, dy, dym1, dz, dzm1, sites) 

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

    def right_bc_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                        dy, dym1, dz, dzm1, sites):
        # bn derivatives -------------------------------------
        defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s,\
        dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN = \
        bnp_derivs('electrons', sys, efn, v, smNN_s, smN_s, s_spN, s_spNN,\
                   dy, dym1, dz, dzm1, sites)

        dbn_rows = np.reshape(np.repeat(3*sites, 13), (len(sites), 13)).tolist()

        dbn_cols = zip(3*smNN_s[0], 3*smNN_s[0]+2, 3*smN_s[0],
                       3*smN_s[0]+2, 3*(sites-1), 3*(sites-1)+2, 3*sites,\
                       3*sites+1, 3*sites+2, 3*s_spN[1], 3*s_spN[1]+2,\
                       3*s_spNN[1], 3*s_spNN[1]+2)

        dbn_data = zip(defn_smNN, dv_smNN, defn_smN, dv_smN, defn_sm1, dv_sm1,\
                       defn_s, defp_s, dv_s, defn_spN, dv_spN, defn_spNN, dv_spNN)

        update(dbn_rows, dbn_cols, dbn_data)

        # bp derivatives -------------------------------------
        defp_smNN, dv_smNN, defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s,\
        dv_s, defp_spN, dv_spN, defp_spNN, dv_spNN = \
        bnp_derivs('holes', sys, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                   dy, dym1, dz, dzm1, sites)

        dbp_rows = np.reshape(np.repeat(3*sites+1, 13), (len(sites), 13)).tolist()

        dbp_cols = zip(3*smNN_s[0]+1, 3*smNN_s[0]+2, 3*smN_s[0]+1,
                       3*smN_s[0]+2, 3*(sites-1)+1, 3*(sites-1)+2, 3*sites,\
                       3*sites+1, 3*sites+2, 3*s_spN[1]+1, 3*s_spN[1]+2,\
                       3*s_spNN[1]+1, 3*s_spNN[1]+2)

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
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    ddp_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
               dx, dxm1, dy, dym1, dz, dzm1, sites)


    ###########################################################################
    #        left boundary: i = 0, 0 <= j <= Ny-1, 0 <= k <= Nz-1             #
    ###########################################################################
    # We compute an, ap, av derivatives. Those functions are only defined on the
    # left boundary of the system.

    # list of the sites on the left side
    sites = [j*Nx + k*Nx*Ny for k in range(Nz) for j in range(Ny)]
    sites = np.asarray(sites)

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

    # dybar and dzbar
    dy = np.repeat(sys.dy[1:], Nz-2)
    dym1 = np.repeat(sys.dy[:-1], Nz-2)
    dz = np.repeat(sys.dz[1:], Ny-2)
    dzm1 = np.repeat(sys.dz[:-1], Ny-2)

    # compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    right_bc_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                    dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #           right boundary: i = Nx-1, j = Ny-1, 0 < k < Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + (Ny-1)*Nx + k*Nx*Ny for k in range(1,Nz-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = np.repeat((sys.dy[0] + sys.dy[-1]) / 2., Nz-2)
    dym1 = np.repeat(sys.dy[-1], Nz-2) 
    dz = sys.dz[1:]
    dzm1 = sys.dz[:-1]

    # compute the currents
    smNN_s =[sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites - Nx*(Ny-1)]
    s_spNN =[sites, sites + Nx*Ny]

    right_bc_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                    dy, dym1, dz, dzm1, sites)

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

    right_bc_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                    dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #           right boundary: i = Nx-1, 0 < j < Ny-1, k = Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + j*Nx + (Nz-1)*Nx*Ny for j in range(1,Ny-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = sys.dy[1:]
    dym1 = sys.dy[:-1]
    dz = np.repeat((sys.dz[-1] + sys.dz[0])/2., Ny-2)
    dzm1 = np.repeat(sys.dz[-1], Ny-2)

    # compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites - Nx*Ny*(Nz-1)]

    right_bc_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                    dy, dym1, dz, dzm1, sites)

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

    right_bc_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                    dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = Ny-1, k = 0              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + (Ny-1)*Nx]
    sites = np.asarray(sites)

    # lattice distances
    dy = (sys.dy[0] + sys.dy[-1])/2.
    dym1 = sys.dy[-1]
    dz = sys.dz[0]
    dzm1 = (sys.dz[-1] + sys.dz[0])/2.

    # compute the currents
    smNN_s = [sites + Nx*Ny*(Nz-1), sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites - Nx*(Ny-1)]
    s_spNN = [sites, sites + Nx*Ny]

    right_bc_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                    dy, dym1, dz, dzm1, sites)

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

    right_bc_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                    dy, dym1, dz, dzm1, sites)

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

    right_bc_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                    dy, dym1, dz, dzm1, sites)

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

    right_bc_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                    dy, dym1, dz, dzm1, sites)


    ###########################################################################
    #            faces between contacts: 0 < i < Nx-1, j or k fixed           #
    ###########################################################################
    # Here we focus on the faces between the contacts. There are 4 cases
    # (obviously).

    dx = np.tile(sys.dx[1:], Ny-2)
    dxm1 = np.tile(sys.dx[:-1], Ny-2)

    ###########################################################################
    #              z-face top: 0 < i < Nx-1, 0 < j < Ny-1, k = Nz-1           #
    ###########################################################################
    # list of the sites
    sites = [i + j*Nx + (Nz-1)*Nx*Ny for j in range(1,Ny-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = np.repeat(sys.dy[1:], Nx-2)
    dz = np.repeat((sys.dz[0] + sys.dz[-1])/2., (Nx-2)*(Ny-2))
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dzm1 = np.repeat(sys.dz[-1], (Nx-2)*(Ny-2))

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites - Nx*Ny*(Nz-1)]

    ddp_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
               dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #             z- face bottom: 0 < i < Nx-1, 0 < j < Ny-1, k = 0           #
    ###########################################################################
    # list of the sites
    sites = [i + j*Nx for j in range(1,Ny-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = np.repeat(sys.dy[1:], Nx-2)
    dz = np.repeat(sys.dz[0], (Nx-2)*(Ny-2))
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dzm1 = np.repeat((sys.dz[0] + sys.dz[-1])/2., (Nx-2)*(Ny-2))

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites + Nx*Ny*(Nz-1), sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    ddp_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
               dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #            y-face front: 0 < i < Nx-1, j = 0, 0 < k < Nz-1              #
    ###########################################################################
    # list of the sites
    sites = [i + k*Nx*Ny for k in range(1,Nz-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Nz-2)
    dxm1 = np.tile(sys.dx[:-1], Nz-2)
    dy = np.repeat(sys.dy[0], (Nx-2)*(Nz-2))
    dz = np.repeat(sys.dz[1:], (Nx-2))
    dym1 = np.repeat((sys.dy[0] + sys.dy[-1])/2., (Nx-2)*(Nz-2))
    dzm1 = np.repeat(sys.dz[:-1], Nx-2)

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites + Nx*(Ny-1), sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    ddp_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
               dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #            y-face back: 0 < i < Nx-1, j = Ny-1, 0 < k < Nz-1            #
    ###########################################################################
    # list of the sites
    sites = [i + (Ny-1)*Nx + k*Nx*Ny for k in range(1,Nz-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = np.repeat((sys.dy[0] + sys.dy[-1])/2., (Nx-2)*(Nz-2))
    dz = np.repeat(sys.dz[1:], Nx-2)
    dym1 = np.repeat(sys.dy[0], (Nx-2)*(Nz-2))
    dzm1 = np.repeat(sys.dz[:-1], Nx-2)

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites - Nx*(Ny-1)]
    s_spNN = [sites, sites + Nx*Ny]

    ddp_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
               dx, dxm1, dy, dym1, dz, dzm1, sites)


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

    ddp_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
               dx, dxm1, dy, dym1, dz, dzm1, sites)


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

    ddp_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
               dx, dxm1, dy, dym1, dz, dzm1, sites)

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

    ddp_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
               dx, dxm1, dy, dym1, dz, dzm1, sites)

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

    ddp_derivs(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
               dx, dxm1, dy, dym1, dz, dzm1, sites)

    if with_mumps:
        J = coo_matrix((data, (rows, columns)), shape=(3*Nx*Ny*Nz, 3*Nx*Ny*Nz), dtype=np.float64)
    else:
        J = csc_matrix((data, (rows, columns)), shape=(3*Nx*Ny*Nz, 3*Nx*Ny*Nz), dtype=np.float64)
    return J
