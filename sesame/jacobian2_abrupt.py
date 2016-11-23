import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from itertools import chain

from sesame.observables import *

def getJ(sys, v, efn, efp, with_mumps):
    ###########################################################################
    #                     organization of the Jacobian matrix                 #
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

    Nx, Ny = sys.xpts.shape[0], sys.ypts.shape[0]

    # lists of rows, columns and data that will create the sparse Jacobian
    global rows, columns, data
    rows = []
    columns = []
    data = []

    ###########################################################################
    #                     For all sites in the system                         #
    ###########################################################################
    sites = [i + j*Nx for j in range(Ny) for i in range(Nx)]

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
        rows.extend(chain.from_iterable(r))
        columns.extend(chain.from_iterable(c))
        data.extend(chain.from_iterable(d))


    def f_derivatives(carriers, djx_s, djx_sm1, djy_s, djy_smN, dxbar, dybar, sites):
        # The function is written with p indices but is valid for both n and p

        # currents derivatives
        djx_s_def_s, djx_s_def_sp1, djx_s_dv_s, djx_s_dv_sp1         = djx_s
        djx_sm1_def_sm1, djx_sm1_def_s, djx_sm1_dv_sm1, djx_sm1_dv_s = djx_sm1
        djy_s_def_s, djy_s_def_spN, djy_s_dv_s, djy_s_dv_spN         = djy_s
        djy_smN_def_smN, djy_smN_def_s, djy_smN_dv_smN, djy_smN_dv_s = djy_smN

        # compute the derivatives of fp and fn
        def_smN = - djy_smN_def_smN / dybar
        dv_smN = - djy_smN_dv_smN / dybar

        def_sm1 = - djx_sm1_def_sm1 / dxbar
        dv_sm1 = - djx_sm1_dv_sm1 / dxbar

        dv_s = (djx_s_dv_s - djx_sm1_dv_s) / dxbar + \
               (djy_s_dv_s - djy_smN_dv_s) / dybar
        if carriers == 'holes':
            defn_s = dr_defn_s[sites]
            defp_s = (djx_s_def_s - djx_sm1_def_s) / dxbar + \
                     (djy_s_def_s - djy_smN_def_s) / dybar + dr_defp_s[sites]
            dv_s = dv_s + dr_dv_s[sites]
        if carriers == 'electrons':
            defn_s = (djx_s_def_s - djx_sm1_def_s) / dxbar + \
                     (djy_s_def_s - djy_smN_def_s) / dybar - dr_defn_s[sites]
            defp_s = - dr_defp_s[sites]
            dv_s = dv_s - dr_dv_s[sites]

        def_sp1 = djx_s_def_sp1 / dxbar
        dv_sp1 = djx_s_dv_sp1 / dxbar

        def_spN = djy_s_def_spN / dybar
        dv_spN = djy_s_dv_spN / dybar

        return def_smN, dv_smN, def_sm1, dv_sm1, defn_s, defp_s, dv_s,\
               def_sp1, dv_sp1, def_spN, dv_spN 

    def fv_derivatives(dx, dy, dxm1, dym1, sites):
        dxbar = (dx + dxm1) / 2
        dybar = (dy + dym1) / 2

        # compute the derivatives
        dvmN = -1./(dym1 * dybar)
        dvm1 = -1./(dxm1 * dxbar)
        dv = 2./(dx * dxm1) + 2./(dy * dym1) - drho_dv_s[sites]
        defn = - drho_defn_s[sites]
        defp = - drho_defp_s[sites]
        dvp1 = -1./(dx * dxbar)
        dvpN = -1./(dy * dybar)
        
        return dvmN, dvm1, dv, defn, defp, dvp1, dvpN

    def bn_derivatives(carriers, djx_sm1, djy_s, djy_smN, dxbar, dybar, sites):
        djx_sm1_def_sm1, djx_sm1_def_s, djx_sm1_dv_sm1, djx_sm1_dv_s = djx_sm1
        djy_s_def_s, djy_s_def_spN, djy_s_dv_s, djy_s_dv_spN         = djy_s
        djy_smN_def_smN, djy_smN_def_s, djy_smN_dv_smN, djy_smN_dv_s = djy_smN

        # compute bn derivatives
        def_smN = dxbar/dybar * djy_smN_def_smN
        dv_smN = dxbar/dybar * djy_smN_dv_smN
        
        def_sm1 = djx_sm1_def_sm1
        dv_sm1 = djx_sm1_dv_sm1

        if carriers == 'electrons':
            defn_s = djx_sm1_def_s + dxbar * (dr_defn_s[sites]\
                   - (djy_s_def_s - djy_smN_def_s) / dybar) + sys.Scn[1] * n[sites]
            defp_s = dxbar * dr_defp_s[sites]
            dv_s = djx_sm1_dv_s + dxbar * (dr_dv_s[sites]\
                   - (djy_s_dv_s  - djy_smN_dv_s) / dybar) + sys.Scn[1] * n[sites]

        if carriers == 'holes':
            defn_s = - dxbar * dr_defn_s[sites]
            defp_s = djx_sm1_def_s + dxbar * (-dr_defp_s[sites]\
                     - (djy_s_def_s - djy_smN_def_s) / dybar) - sys.Scp[1] * p[sites]
            dv_s = djx_sm1_dv_s + dxbar * (-dr_dv_s[sites] \
                   - (djy_s_dv_s  - djy_smN_dv_s) / dybar) + sys.Scp[1] * p[sites]

        def_spN = - dxbar/dybar * djy_s_def_spN
        dv_spN = - dxbar/dybar * djy_s_dv_spN

        return def_smN, dv_smN, def_sm1, dv_sm1, defn_s, defp_s, dv_s,\
               def_spN, dv_spN 


    ###########################################################################
    #       inside the system: 0 < i < Nx-1 and 0 < j < Ny-1                  #
    ###########################################################################
    # We compute fn, fp, fv derivatives. Those functions are only defined on the
    # inner part of the system. All the edges containing boundary conditions.

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

    #------------------------ fn derivatives ----------------------------------
    # get the derivatives of jx_s, jx_sm1, jy_s, jy_smN
    djx_s = get_jn_derivs(sys, efn, v, sites, sites + 1, dx)
    djx_sm1 = get_jn_derivs(sys, efn, v, sites - 1, sites, dxm1)

    djy_s = get_jn_derivs(sys, efn, v, sites, sites + Nx, dy)
    djy_smN = get_jn_derivs(sys, efn, v, sites - Nx, sites, dym1)

    defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s, defn_sp1, dv_sp1,\
    defn_spN, dv_spN = \
    f_derivatives('electrons', djx_s, djx_sm1, djy_s, djy_smN, dxbar, dybar, sites) 

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = np.reshape(np.repeat(3*sites, 11), (len(sites), 11)).tolist()

    dfn_cols = zip(3*(sites-Nx), 3*(sites-Nx)+2, 3*(sites-1), 3*(sites-1)+2,
                   3*sites, 3*sites+1, 3*sites+2, 3*(sites+1), 3*(sites+1)+2,\
                   3*(sites+Nx), 3*(sites+Nx)+2)

    dfn_data = zip(defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defn_sp1, dv_sp1, defn_spN, dv_spN)

    update(dfn_rows, dfn_cols, dfn_data)

    #------------------------ fp derivatives ----------------------------------
    # get the derivatives of jx_s, jx_sm1, jy_s, jy_smN
    djx_s = get_jp_derivs(sys, efp, v, sites, sites + 1, dx)
    djx_sm1 = get_jp_derivs(sys, efp, v, sites - 1, sites, dxm1)

    djy_s = get_jp_derivs(sys, efp, v, sites, sites + Nx, dy)
    djy_smN = get_jp_derivs(sys, efp, v, sites - Nx, sites, dym1)

    defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s, defp_sp1, dv_sp1,\
    defp_spN, dv_spN = \
    f_derivatives('holes', djx_s, djx_sm1, djy_s, djy_smN, dxbar, dybar, sites) 

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = np.reshape(np.repeat(3*sites+1, 11), (len(sites), 11)).tolist()

    dfp_cols = zip(3*(sites-Nx)+1, 3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2,
                   3*sites, 3*sites+1, 3*sites+2, 3*(sites+1)+1, 3*(sites+1)+2,\
                   3*(sites+Nx)+1, 3*(sites+Nx)+2)

    dfp_data = zip(defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defp_sp1, dv_sp1, defp_spN, dv_spN)

    update(dfp_rows, dfp_cols, dfp_data)


    #---------------- fv derivatives inside the system ------------------------
    dvmN, dvm1, dv, defn, defp, dvp1, dvpN = fv_derivatives(dx, dy, dxm1, dym1, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = np.reshape(np.repeat(3*sites+2, 7), (len(sites), 7)).tolist()

    dfv_cols = zip(3*(sites-Nx)+2, 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,
                   3*(sites+1)+2, 3*(sites+Nx)+2)

    dfv_data = zip(dvmN, dvm1, defn, defp, dv, dvp1, dvpN)

    update(dfv_rows, dfv_cols, dfv_data)


    ###########################################################################
    #                 left boundary: i = 0 and 0 <= j <= Ny-1                 #
    ###########################################################################
    # We compute an, ap, av derivatives. Those functions are only defined on the
    # left boundary of the system.

    # list of the sites on the left side
    sites = [j*Nx for j in range(Ny)]
    sites = np.asarray(sites)

    #-------------------------- an derivatives --------------------------------
    defn_s, defn_sp1, dv_s, dv_sp1 = get_jn_derivs(sys, efn, v, sites, sites+1, sys.dx[0])

    defn_s -= sys.Scn[0] * n[sites]
    dv_s -= sys.Scn[0] * n[sites]

    # update the sparse matrix row and columns
    dan_rows = [4*[3*s] for s in sites]

    dan_cols = zip(3*sites, 3*sites+2, 3*(sites+1), 3*(sites+1)+2)

    dan_data = zip(defn_s, dv_s, defn_sp1, dv_sp1)

    update(dan_rows, dan_cols, dan_data)

    #-------------------------- ap derivatives --------------------------------
    defp_s, defp_sp1, dv_s, dv_sp1 = get_jp_derivs(sys, efp, v, sites, sites+1, sys.dx[0])

    defp_s += sys.Scp[0] * p[sites]
    dv_s -= sys.Scp[0] * p[sites]

    # update the sparse matrix row and columns
    dap_rows = [4*[3*s+1] for s in sites]

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
    #                right boundary: i = Nx-1 and 0 < j < Ny-1                #
    ###########################################################################
    # We compute bn, bp, bv derivatives. Those functions are only defined on the
    # right boundary of the system.

    # list of the sites on the right side
    sites = [Nx-1 + j*Nx for j in range(1,Ny-1)]
    sites = np.asarray(sites)

    # dxbar and dybar
    dxm1 = sys.dx[-1]
    dy = sys.dy[1:]
    dym1 = sys.dy[:-1]
    dxbar = np.tile(sys.dx[-1], Ny-2)
    dybar = (dy + dym1) / 2.

    #-------------------------- bn derivatives --------------------------------
    # compute the currents derivatives
    djx_sm1 = get_jn_derivs(sys, efn, v, sites - 1, sites, dxm1)
    djy_s   = get_jn_derivs(sys, efn, v, sites, sites + Nx, dy)
    djy_smN = get_jn_derivs(sys, efn, v, sites - Nx, sites, dym1)

    defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s,\
    defn_spN, dv_spN =\
    bn_derivatives('electrons', djx_sm1, djy_s, djy_smN, dxbar, dybar, sites)

    # update the sparse matrix row and columns
    dbn_rows = [9*[3*s] for s in sites]

    dbn_cols = zip(3*(sites-Nx), 3*(sites-Nx)+2, 3*(sites-1), 3*(sites-1)+2,
                   3*sites, 3*sites+1, 3*sites+2, 3*(sites+Nx), 3*(sites+Nx)+2)

    dbn_data = zip(defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, \
                   dv_s, defn_spN, dv_spN)

    update(dbn_rows, dbn_cols, dbn_data)

    #-------------------------- bp derivatives --------------------------------
    # compute the currents derivatives
    djx_sm1 = get_jp_derivs(sys, efp, v, sites - 1, sites, dxm1)
    djy_s   = get_jp_derivs(sys, efp, v, sites, sites + Nx, dy)
    djy_smN = get_jp_derivs(sys, efp, v, sites - Nx, sites, dym1)

    defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s,\
    defp_spN, dv_spN =\
    bn_derivatives('holes', djx_sm1, djy_s, djy_smN, dxbar, dybar, sites)

    # update the sparse matrix row and columns
    dbp_rows = [9*[3*s+1] for s in sites]

    dbp_cols = zip(3*(sites-Nx)+1, 3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2,
                   3*sites, 3*sites+1, 3*sites+2, 3*(sites+Nx)+1, 3*(sites+Nx)+2)

    dbp_data = zip(defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, \
                   dv_s, defp_spN, dv_spN)

    update(dbp_rows, dbp_cols, dbp_data)

    #-------------------------- bv derivatives --------------------------------
    dbv_rows = (3*sites+2).tolist()
    dbv_cols = (3*sites+2).tolist()
    dbv_data = [1 for s in sites] # dv_s = 0

    rows += dbv_rows
    columns += dbv_cols
    data += dbv_data

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

    #-------------------------- bn derivatives --------------------------------
    # compute the currents derivatives
    djx_sm1 = get_jn_derivs(sys, efn, v, Nx-2, Nx-1, dxm1)
    djy_s   = get_jn_derivs(sys, efn, v, Nx-1, 2*Nx-1, dy)
    djy_smN = [0, 0, 0, 0]

    _, _, defn_sm1, dv_sm1, defn_s, defp_s, dv_s, defn_spN, dv_spN =\
    bn_derivatives('electrons', djx_sm1, djy_s, djy_smN, dxbar, dybar, sites)

    # update the sparse matrix row and columns
    dbn_rows = [7*[3*s] for s in sites]

    dbn_cols = [3*(sites-1), 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,
                3*(sites+Nx), 3*(sites+Nx)+2]

    dbn_data = [defn_sm1, dv_sm1, defn_s[0], defp_s[0], dv_s[0], defn_spN, dv_spN]

    update(dbn_rows, dbn_cols, [dbn_data])

    #-------------------------- bp derivatives --------------------------------
    # compute the currents derivatives
    djx_sm1 = get_jp_derivs(sys, efp, v, Nx-2, Nx-1, dxm1)
    djy_s   = get_jp_derivs(sys, efp, v, Nx-1, 2*Nx-1, dy)
    djy_smN = [0, 0, 0, 0]

    _, _, defp_sm1, dv_sm1, defn_s, defp_s, dv_s, defp_spN, dv_spN =\
    bn_derivatives('holes', djx_sm1, djy_s, djy_smN, dxbar, dybar, sites)

    # update the sparse matrix row and columns
    dbp_rows = [7*[3*s+1] for s in sites]

    dbp_cols = [3*(sites-1)+1, 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,\
                3*(sites+Nx)+1, 3*(sites+Nx)+2]

    dbp_data = [defp_sm1, dv_sm1, defn_s[0], defp_s[0], dv_s[0], defp_spN, dv_spN]

    update(dbp_rows, dbp_cols, [dbp_data])

    #-------------------------- bv derivatives --------------------------------
    dbv_rows = (3*sites+2).tolist()
    dbv_cols = (3*sites+2).tolist()
    dbv_data = [1 for s in sites] # dv_s = 0

    rows += dbv_rows
    columns += dbv_cols
    data += dbv_data

    ###########################################################################
    #                    right boundary: i = Nx-1 and j = Ny-1                #
    ###########################################################################
    # list of the sites
    sites = np.array([Nx*Ny-1])

    # dxbar and dybar
    dxm1 = sys.dx[-1]
    dy = 0
    dym1 = sys.dy[-1]
    dxbar = sys.dx[-1]
    dybar = (dy + dym1) / 2.

    #-------------------------- bn derivatives --------------------------------
    # compute the currents derivatives
    djx_sm1 = get_jn_derivs(sys, efn, v, Nx*Ny-2, Nx*Ny-1, dxm1)
    djy_s   = [0, 0, 0, 0]
    djy_smN = get_jn_derivs(sys, efn, v, Nx*(Ny-1)-1, Nx*Ny-1, dym1)

    defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s, _, _ =\
    bn_derivatives('electrons', djx_sm1, djy_s, djy_smN, dxbar, dybar, sites)

    # update the sparse matrix row and columns
    dbn_rows = [7*[3*s] for s in sites]

    dbn_cols = [3*(sites-Nx), 3*(sites-Nx)+2, 3*(sites-1), 3*(sites-1)+2,\
                3*sites, 3*sites+1, 3*sites+2]

    dbn_data = [defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s[0], defp_s[0], dv_s[0]]

    update(dbn_rows, dbn_cols, [dbn_data])

    #-------------------------- bp derivatives --------------------------------
    # compute the currents derivatives
    djx_sm1 = get_jp_derivs(sys, efp, v, Nx*Ny-2, Nx*Ny-1, dxm1)
    djy_s   = [0, 0, 0, 0]
    djy_smN = get_jp_derivs(sys, efp, v, Nx*(Ny-1)-1, Nx*Ny-1, dym1)

    defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s, _, _ =\
    bn_derivatives('holes', djx_sm1, djy_s, djy_smN, dxbar, dybar, sites)

    # update the sparse matrix row and columns
    dbp_rows = [7*[3*s+1] for s in sites]

    dbp_cols = [3*(sites-Nx)+1, 3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2,\
                3*sites, 3*sites+1, 3*sites+2]

    dbp_data = [defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s[0], defp_s[0], dv_s[0]]

    update(dbp_rows, dbp_cols, [dbp_data])

    #-------------------------- bv derivatives --------------------------------
    dbv_rows = (3*sites+2).tolist()
    dbv_cols = (3*sites+2).tolist()
    dbv_data = [1 for s in sites] # dv_s = 0

    rows += dbv_rows
    columns += dbv_cols
    data += dbv_data


    ###########################################################################
    #                   boundary: 0 < i < Nx-1 and j = 0                      #
    ###########################################################################
    # We apply drift diffusion equations.

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

    #------------------------ fn derivatives ----------------------------------
    # get the derivatives of jx_s, jx_sm1, jy_s, jy_smN
    djx_s = get_jn_derivs(sys, efn, v, sites, sites + 1, dx)
    djx_sm1 = get_jn_derivs(sys, efn, v, sites - 1, sites, dxm1)

    djy_s = get_jn_derivs(sys, efn, v, sites, sites + Nx, dy)
    djy_smN = [0, 0, 0, 0]

    _, _, defn_sm1, dv_sm1, defn_s, defp_s, dv_s, defn_sp1, dv_sp1,\
    defn_spN, dv_spN = \
    f_derivatives('electrons', djx_s, djx_sm1, djy_s, djy_smN, dxbar, dybar, sites) 

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = [9*[3*s] for s in sites]

    dfn_cols = zip(3*(sites-1), 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,
                   3*(sites+1), 3*(sites+1)+2, 3*(sites+Nx), 3*(sites+Nx)+2)

    dfn_data = zip(defn_sm1, dv_sm1, defn_s, defp_s, dv_s, defn_sp1, dv_sp1,\
                   defn_spN, dv_spN)

    update(dfn_rows, dfn_cols, dfn_data)

    #------------------------ fp derivatives ----------------------------------
    # get the derivatives of jx_s, jx_sm1, jy_s, jy_smN
    djx_s = get_jp_derivs(sys, efp, v, sites, sites+1, dx)
    djx_sm1 = get_jp_derivs(sys, efp, v, sites - 1, sites, dxm1)

    djy_s = get_jp_derivs(sys, efp, v, sites, sites+Nx, dy)
    djy_smN = [0, 0, 0, 0]

    _, _, defp_sm1, dv_sm1, defn_s, defp_s, dv_s, defp_sp1, dv_sp1,\
    defp_spN, dv_spN = \
    f_derivatives('holes', djx_s, djx_sm1, djy_s, djy_smN, dxbar, dybar, sites) 

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = [9*[3*s+1] for s in sites]

    dfp_cols = zip(3*(sites-1)+1, 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,
                   3*(sites+1)+1, 3*(sites+1)+2, 3*(sites+Nx)+1, 3*(sites+Nx)+2)

    dfp_data = zip(defp_sm1, dv_sm1, defn_s, defp_s, dv_s, defp_sp1, dv_sp1,\
                   defp_spN, dv_spN)

    update(dfp_rows, dfp_cols, dfp_data)

    #---------------- fv derivatives inside the system ------------------------
    # compute the derivatives
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) + 1./(dy * dybar) - drho_dv_s[sites]
    defn = - drho_defn_s[sites]
    defp = - drho_defp_s[sites]
    dvp1 = -1./(dx * dxbar)
    dvpN = -1./(dy * dybar)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = [6*[3*s+2] for s in sites]

    dfv_cols = zip(3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2, 3*(sites+1)+2,\
                   3*(sites+Nx)+2)

    dfv_data = zip(dvm1, defn, defp, dv, dvp1, dvpN)

    update(dfv_rows, dfv_cols, dfv_data)

    ###########################################################################
    #                  boundary: 0 < i < Nx-1 and j = Ny-1                    #
    ###########################################################################
    # We apply drift diffusion equations.

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

    #------------------------ fn derivatives ----------------------------------
    # get the derivatives of jx_s, jx_sm1, jy_s, jy_smN
    djx_s = get_jn_derivs(sys, efn, v, sites, sites+1, dx)
    djx_sm1 = get_jn_derivs(sys, efn, v, sites-1, sites, dxm1)

    djy_s = [0, 0, 0, 0]
    djy_smN = get_jn_derivs(sys, efn, v, sites-Nx, sites, dym1)

    defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s, defn_sp1, dv_sp1,\
    _, _ = \
    f_derivatives('electrons', djx_s, djx_sm1, djy_s, djy_smN, dxbar, dybar, sites) 

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = [9*[3*s] for s in sites]

    dfn_cols = zip(3*(sites-Nx), 3*(sites-Nx)+2, 3*(sites-1), 3*(sites-1)+2, 3*sites,\
                   3*sites+1, 3*sites+2, 3*(sites+1), 3*(sites+1)+2)

    dfn_data = zip(defn_smN, dv_smN, defn_sm1, dv_sm1,\
                   defn_s, defp_s, dv_s, defn_sp1, dv_sp1)

    update(dfn_rows, dfn_cols, dfn_data)

    #------------------------ fp derivatives ----------------------------------
    # get the derivatives of jx_s, jx_sm1, jy_s, jy_smN
    djx_s = get_jp_derivs(sys, efp, v, sites, sites+1, dx)
    djx_sm1 = get_jp_derivs(sys, efp, v, sites-1, sites, dxm1)

    djy_s = [0, 0, 0, 0]
    djy_smN = get_jp_derivs(sys, efp, v, sites-Nx, sites, dym1)

    defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s, defp_sp1, dv_sp1,\
    _, _ = \
    f_derivatives('holes', djx_s, djx_sm1, djy_s, djy_smN, dxbar, dybar, sites) 

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = [9*[3*s+1] for s in sites]

    dfp_cols = zip(3*(sites-Nx)+1,3*(sites-Nx)+2, 3*(sites-1)+1, 3*(sites-1)+2,\
                   3*sites, 3*sites+1, 3*sites+2, 3*(sites+1)+1, 3*(sites+1)+2)

    dfp_data = zip(defp_smN, dv_smN, defp_sm1, dv_sm1,\
                   defn_s, defp_s, dv_s, defp_sp1, dv_sp1)

    update(dfp_rows, dfp_cols, dfp_data)

    #---------------- fv derivatives inside the system ------------------------
    # compute the derivatives
    dvmN = -1./(dym1 * dybar)
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) + 1./(dym1 * dybar) - drho_dv_s[sites]
    defn = - drho_defn_s[sites]
    defp = - drho_defp_s[sites]
    dvp1 = -1./(dx * dxbar)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = [6*[3*s+2] for s in sites]

    dfv_cols = zip(3*(sites-Nx)+2, 3*(sites-1)+2,
                   3*sites, 3*sites+1, 3*sites+2, 3*(sites+1)+2)

    dfv_data = zip(dvmN, dvm1, defn, defp, dv, dvp1)

    update(dfv_rows, dfv_cols, dfv_data)
    

    if with_mumps == False:
        J = coo_matrix((data, (rows, columns)), shape=(3*Nx*Ny, 3*Nx*Ny), dtype=np.float64)
    else:
        J = csr_matrix((data, (rows, columns)), shape=(3*Nx*Ny, 3*Nx*Ny), dtype=np.float64)
        
    return J
