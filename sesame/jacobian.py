import numpy as np
from itertools import chain

from .observables import *
from .defects  import defectsJ


def getJ(sys, v, efn, efp):
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
    Num = Nx * Ny
    # lists of rows, columns and data that will create the sparse Jacobian
    global rows, columns, data
    rows = []
    columns = []
    data = []

    ###########################################################################
    #                     For all sites in the system                         #
    ###########################################################################
    # carrier densities
    n = sys.Nc * np.exp(+sys.bl + efn + v)
    p = sys.Nv * exp(-sys.Eg - sys.bl - efp - v)

    # bulk charges
    drho_defn_s = - n
    drho_defp_s = - p
    drho_dv_s = - n - p

    # derivatives of the bulk recombination rates
    dr_defn_s, dr_defp_s, dr_dv_s = get_bulk_rr_derivs(sys, n, p)

    # charge defects
    if len(sys.defects_list) != 0:
        defectsJ(sys, sys.defects_list, n, p, drho_dv_s, drho_defn_s, drho_defp_s, dr_defn_s, dr_defp_s, dr_dv_s)

    # reshape the array as array[y-indices, x-indices]
    _sites = np.arange(Nx * Ny, dtype=int).reshape(Ny, Nx)

    def update(r, c, d):
        global rows, columns, data
        rows.extend(chain.from_iterable(r))
        columns.extend(chain.from_iterable(c))
        data.extend(chain.from_iterable(d))


    def f_derivatives(carriers, djx_s, djx_sm1, djy_s, djy_smN, dxbar, dybar, sites):
        # The function is written with p indices but is valid for both n and p

        # currents derivatives
        djx_s_def_s, djx_s_def_sp1, djx_s_dv_s, djx_s_dv_sp1 = djx_s
        djx_sm1_def_sm1, djx_sm1_def_s, djx_sm1_dv_sm1, djx_sm1_dv_s = djx_sm1
        djy_s_def_s, djy_s_def_spN, djy_s_dv_s, djy_s_dv_spN = djy_s
        djy_smN_def_smN, djy_smN_def_s, djy_smN_dv_smN, djy_smN_dv_s = djy_smN

        # compute the derivatives of fp
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

        return def_smN, dv_smN, def_sm1, dv_sm1, defn_s, defp_s, dv_s, \
               def_sp1, dv_sp1, def_spN, dv_spN

    def fv_derivatives(dx, dy, dxm1, dym1, epsilon, sites):

        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.

        infind = np.where(np.isinf(dybar))
        for i in infind[0]:
            if np.isinf(dy[i]):
                dybar[i] = dy[i-Nx] / 2.
            else:
                dybar[i] = dy[i] / 2.

        p1y_ind = np.mod(sites + Nx, Nx * Ny)
        m1y_ind = np.mod(sites - Nx, Nx * Ny)

        eps_m1x = .5 * (epsilon[sites - 1] + epsilon[sites])
        eps_p1x = .5 * (epsilon[sites + 1] + epsilon[sites])
        eps_m1y = .5 * (epsilon[(sites - Nx) % (Nx*Ny)] + epsilon[sites])
        eps_p1y = .5 * (epsilon[(sites + Nx) % (Nx*Ny)] + epsilon[sites])


        dvmN = -eps_m1y * 1. / (dym1 * dybar)
        dvm1 = -eps_m1x * 1. / (dxm1 * dxbar)
        dv = eps_m1x / (dxm1 * dxbar) + eps_p1x / (dx * dxbar) + eps_m1y / (dym1 * dybar) + eps_p1y / (dy * dybar) - \
             drho_dv_s[sites]
        dvp1 = -eps_p1x * 1. / (dx * dxbar)
        dvpN = -eps_p1y * 1. / (dy * dybar)
        defn = - drho_defn_s[sites]
        defp = - drho_defp_s[sites]

        return dvmN, dvm1, dv, defn, defp, dvp1, dvpN


    ###########################################################################
    #       inside the system: 0 < i < Nx-1 and 0 <= j <= Ny-1                #
    ###########################################################################
    # We compute fn, fp, fv derivatives. Those functions are only defined on the
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

    # ------------------------ fn derivatives ----------------------------------
    # get the derivatives of jx_s, jx_sm1, jy_s, jy_smN
    djx_s = get_jn_derivs(sys, efn, v, sites, sites + 1, dx)
    djx_sm1 = get_jn_derivs(sys, efn, v, sites - 1, sites, dxm1)

    djy_s = get_jn_derivs(sys, efn, v, sites, (sites + Nx) % Num , dy)
    djy_smN = get_jn_derivs(sys, efn, v, (sites - Nx) % Num, sites, dym1)

    defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s, defn_sp1, dv_sp1, \
    defn_spN, dv_spN = \
        f_derivatives('electrons', djx_s, djx_sm1, djy_s, djy_smN, dxbar, dybar, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = np.reshape(np.repeat(3 * sites, 11), (len(sites), 11)).tolist()

    dfn_cols = zip(3 * ((sites - Nx)%Num), 3 * ((sites - Nx)%Num) + 2, 3 * (sites - 1), 3 * (sites - 1) + 2,
                   3 * sites, 3 * sites + 1, 3 * sites + 2, 3 * (sites + 1), 3 * (sites + 1) + 2, \
                   3 * ((sites + Nx)%Num), 3 * ((sites + Nx)%Num) + 2)

    dfn_data = zip(defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s, \
                   defn_sp1, dv_sp1, defn_spN, dv_spN)

    update(dfn_rows, dfn_cols, dfn_data)

    # ------------------------ fp derivatives ----------------------------------
    # get the derivatives of jx_s, jx_sm1, jy_s, jy_smN
    djx_s = get_jp_derivs(sys, efp, v, sites, sites + 1, dx)
    djx_sm1 = get_jp_derivs(sys, efp, v, sites - 1, sites, dxm1)

    djy_s = get_jp_derivs(sys, efp, v, sites, (sites + Nx) % Num, dy)
    djy_smN = get_jp_derivs(sys, efp, v, (sites - Nx) % Num, sites, dym1)

    defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s, defp_sp1, dv_sp1, \
    defp_spN, dv_spN = \
        f_derivatives('holes', djx_s, djx_sm1, djy_s, djy_smN, dxbar, dybar, sites)


    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = np.reshape(np.repeat(3 * sites + 1, 11), (len(sites), 11)).tolist()

    dfp_cols = zip(3 * ((sites - Nx)%Num) + 1, 3 * ((sites - Nx)%Num) + 2, 3 * (sites - 1) + 1, 3 * (sites - 1) + 2,
                   3 * sites, 3 * sites + 1, 3 * sites + 2, 3 * (sites + 1) + 1, 3 * (sites + 1) + 2, \
                   3 * ((sites + Nx)%Num) + 1, 3 * ((sites + Nx)%Num) + 2)

    dfp_data = zip(defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s, \
                   defp_sp1, dv_sp1, defp_spN, dv_spN)

    update(dfp_rows, dfp_cols, dfp_data)

    # ---------------- fv derivatives inside the system ------------------------
    dvmN, dvm1, dv, defn, defp, dvp1, dvpN = fv_derivatives(dx, dy, dxm1, dym1, sys.epsilon, sites)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = np.reshape(np.repeat(3 * sites + 2, 7), (len(sites), 7)).tolist()

    dfv_cols = zip(3 * ((sites - Nx)%Num) + 2, 3 * (sites - 1) + 2, 3 * sites, 3 * sites + 1, 3 * sites + 2,
                   3 * (sites + 1) + 2, 3 * ((sites + Nx)%Num) + 2)

    dfv_data = zip(dvmN, dvm1, defn, defp, dv, dvp1, dvpN)

    update(dfv_rows, dfv_cols, dfv_data)

    ###########################################################################
    #                 left boundary: i = 0 and 0 <= j <= Ny-1                 #
    ###########################################################################
    # We compute an, ap, av derivatives. Those functions are only defined on the
    # left boundary of the system.

    # list of the sites on the left side
    sites = _sites[:, 0].flatten()

    # -------------------------- an derivatives --------------------------------
    # s_sp1 = [i for i in zip(sites, sites + 1)]
    defn_s, defn_sp1, dv_s, dv_sp1 = get_jn_derivs(sys, efn, v, sites, sites + 1, sys.dx[0])

    defn_s -= sys.Scn[0] * n[sites]
    dv_s -= sys.Scn[0] * n[sites]

    # update the sparse matrix row and columns
    dan_rows = zip(3 * sites, 3 * sites, 3 * sites, 3 * sites)
    dan_cols = zip(3 * sites, 3 * sites + 2, 3 * (sites + 1), 3 * (sites + 1) + 2)
    dan_data = zip(defn_s, dv_s, defn_sp1, dv_sp1)

    update(dan_rows, dan_cols, dan_data)

    # -------------------------- ap derivatives --------------------------------
    defp_s, defp_sp1, dv_s, dv_sp1 = get_jp_derivs(sys, efp, v, sites, sites + 1, sys.dx[0])
    defp_s -= sys.Scp[0] * p[sites]
    dv_s -= sys.Scp[0] * p[sites]

    # update the sparse matrix row and columns
    dap_rows = zip(3 * sites + 1, 3 * sites + 1, 3 * sites + 1, 3 * sites + 1)
    dap_cols = zip(3 * sites + 1, 3 * sites + 2, 3 * (sites + 1) + 1, 3 * (sites + 1) + 2)
    dap_data = zip(defp_s, dv_s, defp_sp1, dv_sp1)

    update(dap_rows, dap_cols, dap_data)

    # -------------------------- av derivatives --------------------------------
    dav_rows = (3 * sites + 2).tolist()
    dav_cols = (3 * sites + 2).tolist()
    dav_data = np.ones((len(sites, ))).tolist()

    rows += dav_rows
    columns += dav_cols
    data += dav_data

    ###########################################################################
    #                right boundary: i = Nx-1 and 0 <= j <= Ny-1                #
    ###########################################################################
    # We compute bn, bp, bv derivatives. Those functions are only defined on the
    # right boundary of the system.

    # list of the sites on the right side
    sites = _sites[:, Nx - 1].flatten()

    # -------------------------- bn derivatives --------------------------------
    defn_sm1, defn_s, dv_sm1, dv_s = get_jn_derivs(sys, efn, v, sites - 1, sites, sys.dx[-1])
    defn_s += sys.Scn[1] * n[sites]
    dv_s += sys.Scn[1] * n[sites]

    # update the sparse matrix row and columns
    dbn_rows = zip(3 * sites, 3 * sites, 3 * sites, 3 * sites)
    dbn_cols = zip(3 * (sites - 1), 3 * (sites - 1) + 2, 3 * sites, 3 * sites + 2)
    dbn_data = zip(defn_sm1, dv_sm1, defn_s, dv_s)

    update(dbn_rows, dbn_cols, dbn_data)

    # -------------------------- ap derivatives --------------------------------
    defp_sm1, defp_s, dv_sm1, dv_s = get_jp_derivs(sys, efp, v, sites - 1, sites, sys.dx[-1])
    defp_s += sys.Scp[1] * p[sites]
    dv_s += sys.Scp[1] * p[sites]

    # update the sparse matrix row and columns
    dbp_rows = zip(3 * sites + 1, 3 * sites + 1, 3 * sites + 1, 3 * sites + 1)
    dbp_cols = zip(3 * (sites - 1) + 1, 3 * (sites - 1) + 2, 3 * sites + 1, 3 * sites + 2)
    dbp_data = zip(defp_sm1, dv_sm1, defp_s, dv_s)

    update(dbp_rows, dbp_cols, dbp_data)


    # -------------------------- bv derivatives --------------------------------
    dbv_rows = (3 * sites + 2).tolist()
    dbv_cols = (3 * sites + 2).tolist()
    dbv_data = np.ones((len(sites, ))).tolist()  # dv_s = 0

    rows += dbv_rows
    columns += dbv_cols
    data += dbv_data


    return rows, columns, data