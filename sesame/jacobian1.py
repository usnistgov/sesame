# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import numpy as np
from itertools import chain

from .observables import *

def getJ(sys, v, efn, efp):
    ###########################################################################
    #                     organization of the Jacobian matrix                 #
    ###########################################################################
    # A site with coordinates (i) corresponds to a site number s as follows:
    # i = s
    #
    # Rows for (efn_s, efp_s, v_s)
    # ----------------------------
    # fn_row = 3*s
    # fp_row = 3*s+1
    # fv_row = 3*s+2
    #
    # Columns for (efn_s, efp_s, v_s)
    # -------------------------------
    # efn_sm1_col = 3*(s-1)
    # efn_s_col = 3*s
    # efn_sp1_col = 3*(s+1)
    #
    # efp_sm1_col = 3*(s-1)+1
    # efp_s_col = 3*s+1
    # efp_sp1_col = 3*(s+1)+1
    #
    # v_sm1_col = 3*(s-1)+2
    # v_s_col = 3*s+2
    # v_sp1_col = 3*(s+1)+2

    Nx = sys.xpts.shape[0]

    # lists of rows, columns and data that will create the sparse Jacobian
    rows = []
    columns = []
    data = []

    ###########################################################################
    #                     For all sites in the system                         #
    ###########################################################################
    sites = range(Nx)

    # carrier densities
    n = sys.Nc * np.exp(-sys.bl + efn + v)
    p = sys.Nv * exp(-sys.Eg + sys.bl + efp - v)

    # bulk charges
    drho_defn_s = - n
    drho_defp_s = + p
    drho_dv_s = - n - p

    # derivatives of the bulk recombination rates
    dr_defn_s, dr_defp_s, dr_dv_s = get_bulk_rr_derivs(sys, n, p)

    # charge is divided by epsilon
    drho_defn_s = drho_defn_s / sys.epsilon
    drho_defp_s = drho_defp_s / sys.epsilon
    drho_dv_s = drho_dv_s / sys.epsilon

    ###########################################################################
    #                  inside the system: 0 < i < Nx-1                        #
    ###########################################################################
    # We compute fn, fp, fv derivatives. Those functions are only defined on the
    # inner part of the system. All the edges containing boundary conditions.

    # list of the sites inside the system
    sites = np.arange(1,Nx-1)

    # dxbar
    dx = sys.dx[1:]
    dxm1 = sys.dx[:-1]
    dxbar = (dx + dxm1) / 2.

    #--------------------------------------------------------------------------
    #------------------------ fn derivatives ----------------------------------
    #--------------------------------------------------------------------------
    # get the derivatives of jx_s, jx_sm1
    djx_s_defn_s, djx_s_defn_sp1, djx_s_dv_s, djx_s_dv_sp1 = \
    get_jn_derivs(sys, efn, v, sites, sites+1, dx)

    djx_sm1_defn_sm1, djx_sm1_defn_s, djx_sm1_dv_sm1, djx_sm1_dv_s = \
    get_jn_derivs(sys, efn, v, sites-1, sites, dxm1)

    # compute the derivatives of fn
    defn_sm1 = - djx_sm1_defn_sm1 / dxbar
    dv_sm1 = - djx_sm1_dv_sm1 / dxbar

    defn_s = (djx_s_defn_s - djx_sm1_defn_s) / dxbar - dr_defn_s[sites]
    defp_s = - dr_defp_s[sites]
    dv_s = (djx_s_dv_s - djx_sm1_dv_s) / dxbar - dr_dv_s[sites]

    defn_sp1 = djx_s_defn_sp1 / dxbar
    dv_sp1 = djx_s_dv_sp1 / dxbar

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = zip(3*sites, 3*sites,  3*sites, 3*sites, 3*sites, 3*sites, 3*sites)

    dfn_cols = zip(3*(sites-1), 3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2,\
                3*(sites+1), 3*(sites+1)+2)

    dfn_data = zip(defn_sm1, dv_sm1, defn_s, defp_s, dv_s, defn_sp1, dv_sp1)

    rows.extend(chain.from_iterable(dfn_rows))
    columns.extend(chain.from_iterable(dfn_cols))
    data.extend(chain.from_iterable(dfn_data))

    #--------------------------------------------------------------------------
    #------------------------ fp derivatives ----------------------------------
    #--------------------------------------------------------------------------
    # get the derivatives of jx_s, jx_sm1
    djx_s_defp_s, djx_s_defp_sp1, djx_s_dv_s, djx_s_dv_sp1 = \
    get_jp_derivs(sys, efp, v, sites, sites+1, dx)

    djx_sm1_defp_sm1, djx_sm1_defp_s, djx_sm1_dv_sm1, djx_sm1_dv_s = \
    get_jp_derivs(sys, efp, v, sites-1, sites, dxm1)

    # compute the derivatives of fp
    defp_sm1 = - djx_sm1_defp_sm1 / dxbar
    dv_sm1 = - djx_sm1_dv_sm1 / dxbar

    defn_s = dr_defn_s[sites]
    defp_s = (djx_s_defp_s - djx_sm1_defp_s) / dxbar + dr_defp_s[sites]
    dv_s = (djx_s_dv_s - djx_sm1_dv_s) / dxbar + dr_dv_s[sites]

    defp_sp1 = djx_s_defp_sp1 / dxbar
    dv_sp1 = djx_s_dv_sp1 / dxbar

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = zip(3*sites+1, 3*sites+1,  3*sites+1, 3*sites+1, 3*sites+1,
                   3*sites+1, 3*sites+1)

    dfp_cols = zip(3*(sites-1)+1, 3*(sites-1)+2, 3*sites, 3*sites+1,\
                 3*sites+2, 3*(sites+1)+1, 3*(sites+1)+2)

    dfp_data = zip( defp_sm1, dv_sm1, defn_s, defp_s, dv_s, defp_sp1, dv_sp1)

    rows.extend(chain.from_iterable(dfp_rows))
    columns.extend(chain.from_iterable(dfp_cols))
    data.extend(chain.from_iterable(dfp_data))

    #--------------------------------------------------------------------------
    #---------------- fv derivatives inside the system ------------------------
    #--------------------------------------------------------------------------
    # compute the derivatives
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) - drho_dv_s[sites]
    defn = - drho_defn_s[sites]
    defp = - drho_defp_s[sites]
    dvp1 = -1./(dx * dxbar)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = zip(3*sites+2, 3*sites+2,  3*sites+2, 3*sites+2, 3*sites+2)

    dfv_cols = zip(3*(sites-1)+2, 3*sites, 3*sites+1, 3*sites+2, 3*(sites+1)+2)

    dfv_data = zip(dvm1, defn, defp, dv, dvp1)

    rows.extend(chain.from_iterable(dfv_rows))
    columns.extend(chain.from_iterable(dfv_cols))
    data.extend(chain.from_iterable(dfv_data))


    ###########################################################################
    #                           left boundary: i = 0                          #
    ###########################################################################
    # We compute an, ap, av derivatives. Those functions are only defined on the
    # left boundary of the system.

    #--------------------------------------------------------------------------
    #-------------------------- an derivatives --------------------------------
    #--------------------------------------------------------------------------
    defn_s, defn_sp1, dv_s, dv_sp1 = get_jn_derivs(sys, efn, v, 0, 1, sys.dx[0])

    defn_s -= sys.Scn[0] * n[0]
    dv_s -= sys.Scn[0] * n[0]

    # update the sparse matrix row and columns
    dan_rows = [0, 0, 0, 0]
    dan_cols = [0, 2, 3, 3+2]
    dan_data = [defn_s, dv_s, defn_sp1, dv_sp1]

    rows += dan_rows
    columns += dan_cols
    data += dan_data

    #--------------------------------------------------------------------------
    #-------------------------- ap derivatives --------------------------------
    #--------------------------------------------------------------------------
    defp_s, defp_sp1, dv_s, dv_sp1 = get_jp_derivs(sys, efp, v, 0, 1, sys.dx[0])

    defp_s += sys.Scp[0] * p[0]
    dv_s -= sys.Scp[0] * p[0]

    # update the sparse matrix row and columns
    dap_rows = [1, 1, 1, 1]
    dap_cols = [1, 2, 3+1, 3+2]
    dap_data = [defp_s, dv_s, defp_sp1, dv_sp1]

    rows += dap_rows
    columns += dap_cols
    data += dap_data

    #--------------------------------------------------------------------------
    #-------------------------- av derivatives --------------------------------
    #--------------------------------------------------------------------------
    dav_rows = [2]
    dav_cols = [2]
    dav_data = [1]

    rows += dav_rows
    columns += dav_cols
    data += dav_data

    ###########################################################################
    #                       right boundary: i = Nx-1                          #
    ###########################################################################
    # We compute bn, bp, bv derivatives. Those functions are only defined on the
    # right boundary of the system.

    # dxbar
    dxm1 = sys.dx[-1]
    dxbar = sys.dx[-1]

    #--------------------------------------------------------------------------
    #-------------------------- bn derivatives --------------------------------
    #--------------------------------------------------------------------------
    # compute the currents derivatives
    djnx_sm1_defn_sm1, djnx_sm1_defn_s, djnx_sm1_dv_sm1, djnx_sm1_dv_s =\
    get_jn_derivs(sys, efn, v, Nx-2, Nx-1, dxm1)

    # compute bn derivatives
    defn_sm1 = djnx_sm1_defn_sm1
    dv_sm1 = djnx_sm1_dv_sm1

    defn_s = djnx_sm1_defn_s + dxbar * dr_defn_s[Nx-1] + sys.Scn[1] * n[Nx-1]
    defp_s = dxbar * dr_defp_s[Nx-1]
    dv_s = djnx_sm1_dv_s + dxbar * dr_dv_s[Nx-1] + sys.Scn[1] * n[Nx-1]

    # update the sparse matrix row and columns
    dbn_rows = [3*(Nx-1), 3*(Nx-1), 3*(Nx-1), 3*(Nx-1), 3*(Nx-1)]
    dbn_cols = [3*(Nx-1-1), 3*(Nx-1-1)+2, 3*(Nx-1), 3*(Nx-1)+1, 3*(Nx-1)+2]
    dbn_data = [defn_sm1, dv_sm1, defn_s, defp_s, dv_s]

    rows += dbn_rows
    columns += dbn_cols
    data += dbn_data

    #--------------------------------------------------------------------------
    #-------------------------- bp derivatives --------------------------------
    #--------------------------------------------------------------------------
    # compute the currents derivatives
    djpx_sm1_defp_sm1, djpx_sm1_defp_s, djpx_sm1_dv_sm1, djpx_sm1_dv_s =\
    get_jp_derivs(sys, efp, v, Nx-2, Nx-1, dxm1)

    # compute bp derivatives
    defp_sm1 = djpx_sm1_defp_sm1
    dv_sm1 = djpx_sm1_dv_sm1

    defn_s = - dxbar * dr_defn_s[Nx-1]
    defp_s = djpx_sm1_defp_s - dxbar * dr_defp_s[Nx-1] - sys.Scp[1] * p[Nx-1]
    dv_s = djpx_sm1_dv_s - dxbar * dr_dv_s[Nx-1] + sys.Scp[1] * p[Nx-1]

    # update the sparse matrix row and columns
    dbp_rows = [3*(Nx-1)+1, 3*(Nx-1)+1, 3*(Nx-1)+1, 3*(Nx-1)+1, 3*(Nx-1)+1]
    dbp_cols = [3*(Nx-1-1)+1, 3*(Nx-1-1)+2, 3*(Nx-1), 3*(Nx-1)+1, 3*(Nx-1)+2]
    dbp_data = [defp_sm1, dv_sm1, defn_s, defp_s, dv_s]

    rows += dbp_rows
    columns += dbp_cols
    data += dbp_data

    #--------------------------------------------------------------------------
    #-------------------------- bv derivatives --------------------------------
    #--------------------------------------------------------------------------
    dbv_rows = [3*(Nx-1)+2]
    dbv_cols = [3*(Nx-1)+2]
    dbv_data = [1] # dv_s = 0

    rows += dbv_rows
    columns += dbv_cols
    data += dbv_data

    return rows, columns, data
