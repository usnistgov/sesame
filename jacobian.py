import numpy as np
from numpy import exp
from scipy.sparse import coo_matrix
from itertools import chain

from sesame.observables import *

def getJ(v, efn, efp, params):

    bl, eg, nC, nV, nA, nD, scn, scp, g, mu, tau, rho,\
    NGB, SGB, nGB, pGB,\
    n1, p1, ni, xpts, ypts = params

    Nx = xpts.shape[0]
    Ny = ypts.shape[0]

    delta_x = xpts[1:] - xpts[:-1]
    delta_y = ypts[1:] - ypts[:-1]

    # expand dx and dy in y and x respectively
    dx = np.tile(delta_x, (Ny, 1)).T
    dy = np.tile(delta_y, (Nx, 1))

    # reshape the vectors to conform to [x,y] coordinates
    v_xy = v.reshape(Ny, Nx).T
    efn_xy = efn.reshape(Ny, Nx).T
    efp_xy = efp.reshape(Ny, Nx).T
    g_xy = g.reshape(Ny, Nx).T
    mu = mu.reshape(Ny, Nx).T
    S_xy = (1/tau).reshape(Ny, Nx).T
    rho_xy = rho.reshape(Ny, Nx).T
    SGB_xy = SGB.reshape(Ny, Nx).T
    NGB_xy = NGB.reshape(Ny, Nx).T
    n_xy = get_n(efn_xy, v_xy, params)
    p_xy = get_p(efp_xy, v_xy, params)

    # derivatives of the recombination rates
    dr_defp_s, dr_defn_s, dr_dv_s = \
    get_rr_derivs(n_xy, p_xy, n1, p1, S_xy, params)\

    drGB_defp_s, drGB_defn_s, drGB_dv_s = \
    get_rr_derivs(n_xy, p_xy, nGB, pGB, SGB_xy, params)\

    # GB charge density derivatives
    drhoGB_dv = -NGB_xy * (n_xy*(n_xy+p_xy+nGB+pGB)-(n_xy+pGB)*(n_xy-p_xy))\
                        / (n_xy+p_xy+nGB+pGB)**2
    drhoGB_defn = -NGB_xy * (n_xy*(n_xy+p_xy+nGB+pGB)-(n_xy+pGB)*n_xy)\
                          / (n_xy+p_xy+nGB+pGB)**2
    drhoGB_defp = NGB_xy * (n_xy+pGB)*p_xy / (n_xy+p_xy+nGB+pGB)**2


    # lists of rows, columns and data that will create the sparse Jacobian
    rows = []
    columns = []
    data = []

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

    ###########################################################################
    #       inside the system: 0 < i < Nx-1 and 0 < j < Ny-1                  #
    ###########################################################################
    # We compute fn, fp, fv derivatives. Those functions are only defined on the
    # inner part of the system. All the edges containing boundary conditions.

    # list of the sites inside the system
    sites = [i + j*Nx for i in range(1,Nx-1) for j in range(1,Ny-1)]

    # dxbar and dybar
    dxbar = (dx[1:,1:-1] + dx[:-1,1:-1]) / 2.
    dybar = (dy[1:-1,1:] + dy[1:-1,:-1]) / 2.

    # gether efn, efp, v for all relevant sites
    efn_smN = efn_xy[1:-1,:-2]
    efn_sm1 = efn_xy[:-2,1:-1]
    efn_s = efn_xy[1:-1,1:-1]
    efn_sp1 = efn_xy[2:,1:-1]
    efn_spN = efn_xy[1:-1,2:]

    efp_smN = efp_xy[1:-1,:-2]
    efp_sm1 = efp_xy[:-2,1:-1]
    efp_s = efp_xy[1:-1,1:-1]
    efp_sp1 = efp_xy[2:,1:-1]
    efp_spN = efp_xy[1:-1,2:]

    v_smN = v_xy[1:-1,:-2]
    v_sm1 = v_xy[:-2,1:-1]
    v_s = v_xy[1:-1,1:-1]
    v_sp1 = v_xy[2:,1:-1]
    v_spN = v_xy[1:-1,2:]


    #--------------------------------------------------------------------------
    #------------------------ fn derivatives ----------------------------------
    #--------------------------------------------------------------------------
    # get the derivatives of jx_s, jx_sm1, jy_s, jy_smN
    djx_s_defn_s, djx_s_defn_sp1, djx_s_dv_s, djx_s_dv_sp1 = \
    mu[1:-1,1:-1] * get_jn_derivs(efn_s, efn_sp1, v_s, v_sp1, dx[1:,1:-1], params)

    djx_sm1_defn_sm1, djx_sm1_defn_s, djx_sm1_dv_sm1, djx_sm1_dv_s = \
    mu[:-2,1:-1] * get_jn_derivs(efn_sm1, efn_s, v_sm1, v_s, dx[:-1,1:-1], params)

    djy_s_defn_s, djy_s_defn_spN, djy_s_dv_s, djy_s_dv_spN = \
    mu[1:-1,1:-1] * get_jn_derivs(efn_s, efn_spN, v_s, v_spN, dy[1:-1,1:], params)

    djy_smN_defn_smN, djy_smN_defn_s, djy_smN_dv_smN, djy_smN_dv_s = \
    mu[1:-1,:-2] * get_jn_derivs(efn_smN, efn_s, v_smN, v_s, dy[1:-1,:-1], params)

    # compute the derivatives of fn
    defn_smN = - djy_smN_defn_smN / dybar
    dv_smN = - djy_smN_dv_smN / dybar

    defn_sm1 = - djx_sm1_defn_sm1 / dxbar
    dv_sm1 = - djx_sm1_dv_sm1 / dxbar

    defn_s = (djx_s_defn_s - djx_sm1_defn_s) / dxbar + \
             (djy_s_defn_s - djy_smN_defn_s) / dybar - dr_defn_s[1:-1,1:-1]\
             - drGB_defn_s[1:-1,1:-1]
    defp_s = - dr_defp_s[1:-1,1:-1] - drGB_defp_s[1:-1,1:-1]
    dv_s = (djx_s_dv_s - djx_sm1_dv_s) / dxbar + \
           (djy_s_dv_s - djy_smN_dv_s) / dybar - dr_dv_s[1:-1,1:-1]\
           - drGB_dv_s[1:-1,1:-1]
           
    defn_sp1 = djx_s_defn_sp1 / dxbar
    dv_sp1 = djx_s_dv_sp1 / dxbar

    defn_spN = djy_s_defn_spN / dybar
    dv_spN = djy_s_dv_spN / dybar

    # reshape the derivatives as 1D arrays
    defn_smN = (defn_smN.T).reshape((Nx-2)*(Ny-2))
    dv_smN   = (dv_smN.T).reshape((Nx-2)*(Ny-2))
    defn_sm1 = (defn_sm1.T).reshape((Nx-2)*(Ny-2))
    dv_sm1   = (dv_sm1.T).reshape((Nx-2)*(Ny-2))
    defn_s   = (defn_s.T).reshape((Nx-2)*(Ny-2))
    defp_s   = (defp_s.T).reshape((Nx-2)*(Ny-2))
    dv_s     = (dv_s.T).reshape((Nx-2)*(Ny-2))
    defn_sp1 = (defn_sp1.T).reshape((Nx-2)*(Ny-2))
    dv_sp1   = (dv_sp1.T).reshape((Nx-2)*(Ny-2))
    defn_spN = (defn_spN.T).reshape((Nx-2)*(Ny-2))
    dv_spN   = (dv_spN.T).reshape((Nx-2)*(Ny-2))

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = [11*[3*s] for s in sites]

    dfn_cols = [[3*(s-Nx), 3*(s-Nx)+2, 3*(s-1), 3*(s-1)+2, 3*s, 3*s+1, 3*s+2,\
                3*(s+1), 3*(s+1)+2, 3*(s+Nx), 3*(s+Nx)+2] for s in sites]

    dfn_data = zip(defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defn_sp1, dv_sp1, defn_spN, dv_spN)

    rows += list(chain.from_iterable(dfn_rows))
    columns += list(chain.from_iterable(dfn_cols))
    data += list(chain.from_iterable(dfn_data))

    #--------------------------------------------------------------------------
    #------------------------ fp derivatives ----------------------------------
    #--------------------------------------------------------------------------
    # get the derivatives of jx_s, jx_sm1, jy_s, jy_smN
    djx_s_defp_s, djx_s_defp_sp1, djx_s_dv_s, djx_s_dv_sp1 = \
    mu[1:-1,1:-1] * get_jp_derivs(efp_s, efp_sp1, v_s, v_sp1, dx[1:,1:-1], params)

    djx_sm1_defp_sm1, djx_sm1_defp_s, djx_sm1_dv_sm1, djx_sm1_dv_s = \
    mu[:-2,1:-1] * get_jp_derivs(efp_sm1, efp_s, v_sm1, v_s, dx[:-1,1:-1], params)

    djy_s_defp_s, djy_s_defp_spN, djy_s_dv_s, djy_s_dv_spN = \
    mu[1:-1,1:-1] * get_jp_derivs(efp_s, efp_spN, v_s, v_spN, dy[1:-1,1:], params)

    djy_smN_defp_smN, djy_smN_defp_s, djy_smN_dv_smN, djy_smN_dv_s = \
    mu[1:-1,:-2] * get_jp_derivs(efp_smN, efp_s, v_smN, v_s, dy[1:-1,:-1], params)

    # compute the derivatives of fp
    defp_smN = - djy_smN_defp_smN / dybar
    dv_smN = - djy_smN_dv_smN / dybar

    defp_sm1 = - djx_sm1_defp_sm1 / dxbar
    dv_sm1 = - djx_sm1_dv_sm1 / dxbar

    defn_s = dr_defn_s[1:-1,1:-1] + drGB_defn_s[1:-1,1:-1]
    defp_s = (djx_s_defp_s - djx_sm1_defp_s) / dxbar + \
             (djy_s_defp_s - djy_smN_defp_s) / dybar + dr_defp_s[1:-1,1:-1]\
             + drGB_defp_s[1:-1,1:-1]
    dv_s = (djx_s_dv_s - djx_sm1_dv_s) / dxbar + \
           (djy_s_dv_s - djy_smN_dv_s) / dybar - dr_dv_s[1:-1,1:-1]\
           - drGB_dv_s[1:-1,1:-1]
           
    defp_sp1 = djx_s_defp_sp1 / dxbar
    dv_sp1 = djx_s_dv_sp1 / dxbar

    defp_spN = djy_s_defp_spN / dybar
    dv_spN = djy_s_dv_spN / dybar

    # reshape the derivatives as 1D arrays
    defp_smN = (defp_smN.T).reshape((Nx-2)*(Ny-2))
    dv_smN   = (dv_smN.T).reshape((Nx-2)*(Ny-2))
    defp_sm1 = (defp_sm1.T).reshape((Nx-2)*(Ny-2))
    dv_sm1   = (dv_sm1.T).reshape((Nx-2)*(Ny-2))
    defn_s   = (defn_s.T).reshape((Nx-2)*(Ny-2))
    defp_s   = (defp_s.T).reshape((Nx-2)*(Ny-2))
    dv_s     = (dv_s.T).reshape((Nx-2)*(Ny-2))
    defp_sp1 = (defp_sp1.T).reshape((Nx-2)*(Ny-2))
    dv_sp1   = (dv_sp1.T).reshape((Nx-2)*(Ny-2))
    defp_spN = (defp_spN.T).reshape((Nx-2)*(Ny-2))
    dv_spN   = (dv_spN.T).reshape((Nx-2)*(Ny-2))

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = [11*[3*s+1] for s in sites]

    dfp_cols = [[3*(s-Nx)+1, 3*(s-Nx)+2, 3*(s-1)+1, 3*(s-1)+2, 3*s, 3*s+1,\
                 3*s+2, 3*(s+1)+1, 3*(s+1)+2, 3*(s+Nx)+1, 3*(s+Nx)+2] 
                 for s in sites]

    dfp_data = zip(defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                   defp_sp1, dv_sp1, defp_spN, dv_spN)

    rows += list(chain.from_iterable(dfp_rows))
    columns += list(chain.from_iterable(dfp_cols))
    data += list(chain.from_iterable(dfp_data))

    #--------------------------------------------------------------------------
    #---------------- fv derivatives inside the system ------------------------
    #--------------------------------------------------------------------------
    # compute the derivatives
    dvmN = -1./(dy[1:-1,:-1] * dybar)
    dvm1 = -1./(dx[:-1,1:-1] * dxbar)
    dv = 2./(dx[1:,1:-1] * dx[:-1,1:-1]) + 2./(dy[1:-1,1:] * dy[1:-1,:-1])\
         + p_xy[1:-1,1:-1] + n_xy[1:-1,1:-1] - drhoGB_dv[1:-1,1:-1]
    defn = n_xy[1:-1,1:-1] - drhoGB_defn[1:-1,1:-1]
    defp = -p_xy[1:-1,1:-1] - drhoGB_defp[1:-1,1:-1]
    dvp1 = -1./(dx[1:,1:-1] * dxbar)
    dvpN = -1./(dy[1:-1,1:] * dybar)

    # reshape the derivatives as 1D arrays
    dvmN = (dvmN.T).reshape((Nx-2)*(Ny-2)) 
    dvm1 = (dvm1.T).reshape((Nx-2)*(Ny-2))
    dv   = (dv.T).reshape((Nx-2)*(Ny-2))
    defn = (defn.T).reshape((Nx-2)*(Ny-2))
    defp = (defp.T).reshape((Nx-2)*(Ny-2))
    dvp1 = (dvp1.T).reshape((Nx-2)*(Ny-2))
    dvpN = (dvpN.T).reshape((Nx-2)*(Ny-2))

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = [7*[3*s+2] for s in sites]

    dfv_cols = [[3*(s-Nx)+2, 3*(s-1)+2, 3*s, 3*s+1, 3*s+2, 3*(s+1)+2, 3*(s+Nx)+2]
                for s in sites]

    dfv_data = zip(dvmN, dvm1, defn, defp, dv, dvp1, dvpN)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))


    ###########################################################################
    #                  left boundary: i = 0 and 0 < j < Ny-1                  #
    ###########################################################################
    # We compute an, ap, av derivatives. Those functions are only defined on the
    # left boundary of the system.

    # list of the sites on the left side
    sites = [j*Nx for j in range(1,Ny-1)]

    #--------------------------------------------------------------------------
    #-------------------------- an derivatives --------------------------------
    #--------------------------------------------------------------------------
    defn_s, defn_sp1, dv_s, dv_sp1 = \
    mu[0,1:-1] * get_jn_derivs(efn_xy[0,1:-1], efn_xy[1,1:-1], v_xy[0,1:-1], \
                               v_xy[1,1:-1], dx[0,1:-1], params)

    defn_s -= scn[0] * n_xy[0,1:-1]
    dv_s -= scn[0] * n_xy[0,1:-1]

    # reshape the derivatives as 1D arrays
    defn_s   = (defn_s.T).reshape(Ny-2)
    dv_s     = (dv_s.T).reshape(Ny-2)
    defn_sp1 = (defn_sp1.T).reshape(Ny-2)
    dv_sp1   = (dv_sp1.T).reshape(Ny-2)

    # update the sparse matrix row and columns
    dan_rows = [4*[3*s] for s in sites]

    dan_cols = [[3*s, 3*s+2, 3*(s+1), 3*(s+1)+2] for s in sites]

    dan_data = zip(defn_s, dv_s, defn_sp1, dv_sp1)

    rows += list(chain.from_iterable(dan_rows))
    columns += list(chain.from_iterable(dan_cols))
    data += list(chain.from_iterable(dan_data))

    #--------------------------------------------------------------------------
    #-------------------------- ap derivatives --------------------------------
    #--------------------------------------------------------------------------
    defp_s, defp_sp1, dv_s, dv_sp1 = \
    mu[0,1:-1] * get_jp_derivs(efp_xy[0,1:-1], efp_xy[1,1:-1], v_xy[0,1:-1], \
                               v_xy[1,1:-1], dx[0,1:-1], params)

    defp_s += scp[0] * p_xy[0,1:-1]
    dv_s -= scp[0] * p_xy[0,1:-1]

    # reshape the derivatives as 1D arrays
    defp_s   = (defp_s.T).reshape(Ny-2)
    dv_s     = (dv_s.T).reshape(Ny-2)
    defp_sp1 = (defp_sp1.T).reshape(Ny-2)
    dv_sp1   = (dv_sp1.T).reshape(Ny-2)

    # update the sparse matrix row and columns
    dap_rows = [4*[3*s+1] for s in sites]

    dap_cols = [[3*s+1, 3*s+2, 3*(s+1)+1, 3*(s+1)+2] for s in sites]

    dap_data = zip(defp_s, dv_s, defp_sp1, dv_sp1)

    rows += list(chain.from_iterable(dap_rows))
    columns += list(chain.from_iterable(dap_cols))
    data += list(chain.from_iterable(dap_data))

    #--------------------------------------------------------------------------
    #-------------------------- av derivatives --------------------------------
    #--------------------------------------------------------------------------
    dav_rows = [3*s+2 for s in sites]
    dav_cols = [3*s+2 for s in sites]
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

    # dxbar and dybar
    dxbar = dx[-1,1:-1]
    dybar = (dy[-1,1:] + dy[-1,:-1]) / 2.

    # gather efn, efp, v
    v_sm1 = v_xy[-2,1:-1] 
    v_smN = v_xy[-1,:-2] 
    v_s = v_xy[-1,1:-1]
    v_spN = v_xy[-1,2:] 

    efn_sm1 = efn_xy[-2,1:-1] 
    efn_smN = efn_xy[-1,:-2] 
    efn_s = efn_xy[-1,1:-1]
    efn_spN = efn_xy[-1,2:] 

    efp_sm1 = efp_xy[-2,1:-1] 
    efp_smN = efp_xy[-1,:-2] 
    efp_s = efp_xy[-1,1:-1]
    efp_spN = efp_xy[-1,2:] 

    #--------------------------------------------------------------------------
    #-------------------------- bn derivatives --------------------------------
    #--------------------------------------------------------------------------
    # compute the currents derivatives
    djnx_sm1_defn_sm1, djnx_sm1_defn_s, djnx_sm1_dv_sm1, djnx_sm1_dv_s =\
    mu[-2,1:-1] * get_jn_derivs(efn_sm1, efn_s, v_sm1, v_s, dx[-1,1:-1], params)

    djny_s_defn_s, djny_s_defn_spN, djny_s_dv_s, djny_s_dv_spN = \
    mu[-1,1:-1] * get_jn_derivs(efn_s, efn_spN, v_s, v_spN, dy[-1,1:], params)

    djny_smN_defn_smN, djny_smN_defn_s, djny_smN_dv_smN, djny_smN_dv_s = \
    mu[-1,:-2] * get_jn_derivs(efn_smN, efn_s, v_smN, v_s, dy[-1,:-1], params)

    # compute bn derivatives
    defn_smN = dxbar/dybar * djny_smN_defn_smN
    dv_smN = dxbar/dybar * djny_smN_dv_smN
    
    defn_sm1 = djnx_sm1_defn_sm1
    dv_sm1 = djnx_sm1_dv_sm1

    defn_s = djnx_sm1_defn_s + dxbar * (dr_defn_s[-1,1:-1] + drGB_defn_s[-1,1:-1]\
             - (djny_s_defn_s - djny_smN_defn_s) / dybar) + scn[1] * n_xy[-1,1:-1]
    defp_s = dxbar * (dr_defp_s[-1,1:-1] + drGB_defp_s[-1,1:-1])
    dv_s = djnx_sm1_dv_s + dxbar * (dr_dv_s[-1,1:-1] + drGB_defp_s[-1,1:-1] \
           - (djny_s_dv_s  - djny_smN_dv_s) / dybar) + scn[1] * n_xy[-1,1:-1]

    defn_spN = - dxbar/dybar * djny_s_defn_spN
    dv_spN = - dxbar/dybar * djny_s_dv_spN

    # update the sparse matrix row and columns
    dbn_rows = [9*[3*s] for s in sites]

    dbn_cols = [[3*(s-Nx), 3*(s-Nx)+2, 3*(s-1), 3*(s-1)+2, 3*s, 3*s+1, 3*s+2,\
                 3*(s+Nx), 3*(s+Nx)+2] for s in sites]

    dbn_data = zip(defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, \
                   dv_s, defn_spN, dv_spN)

    rows += list(chain.from_iterable(dbn_rows))
    columns += list(chain.from_iterable(dbn_cols))
    data += list(chain.from_iterable(dbn_data))

    #--------------------------------------------------------------------------
    #-------------------------- bp derivatives --------------------------------
    #--------------------------------------------------------------------------
    # compute the currents derivatives
    djpx_sm1_defp_sm1, djpx_sm1_defp_s, djpx_sm1_dv_sm1, djpx_sm1_dv_s =\
    mu[-2,1:-1] * get_jp_derivs(efp_sm1, efp_s, v_sm1, v_s, dx[-1,1:-1], params)

    djpy_s_defp_s, djpy_s_defp_spN, djpy_s_dv_s, djpy_s_dv_spN = \
    mu[-1,1:-1] * get_jp_derivs(efp_s, efp_spN, v_s, v_spN, dy[-1,1:], params)

    djpy_smN_defp_smN, djpy_smN_defp_s, djpy_smN_dv_smN, djpy_smN_dv_s = \
    mu[-1,:-2] * get_jp_derivs(efp_smN, efp_s, v_smN, v_s, dy[-1,:-1], params)

    # compute bn derivatives
    defp_smN = dxbar/dybar * djpy_smN_defp_smN
    dv_smN = dxbar/dybar * djpy_smN_dv_smN
    
    defp_sm1 = djpx_sm1_defp_sm1
    dv_sm1 = djpx_sm1_dv_sm1

    defn_s = - dxbar * (dr_defn_s[-1,1:-1] + drGB_defn_s[-1,1:-1])
    defp_s = djpx_sm1_defp_s + dxbar * (-dr_defp_s[-1,1:-1] - drGB_defn_s[-1,1:-1]\
             - (djpy_s_defp_s - djpy_smN_defp_s) / dybar) - scp[1] * p_xy[-1,1:-1]
    dv_s = djpx_sm1_dv_s + dxbar * (-dr_dv_s[-1,1:-1] - drGB_defp_s[-1,1:-1] \
           - (djpy_s_dv_s  - djpy_smN_dv_s) / dybar) + scp[1] * p_xy[-1,1:-1]

    defp_spN = - dxbar/dybar * djpy_s_defp_spN
    dv_spN = - dxbar/dybar * djpy_s_dv_spN

    # update the sparse matrix row and columns
    dbp_rows = [9*[3*s+1] for s in sites]

    dbp_cols = [[3*(s-Nx)+1, 3*(s-Nx)+2, 3*(s-1)+1, 3*(s-1)+2, 3*s, 3*s+1, 3*s+2,\
                 3*(s+Nx)+1, 3*(s+Nx)+2] for s in sites]

    dbp_data = zip(defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, \
                   dv_s, defp_spN, dv_spN)

    rows += list(chain.from_iterable(dbp_rows))
    columns += list(chain.from_iterable(dbp_cols))
    data += list(chain.from_iterable(dbp_data))

    #--------------------------------------------------------------------------
    #-------------------------- bv derivatives --------------------------------
    #--------------------------------------------------------------------------
    dbv_rows = [3*s+2 for s in sites]
    dbv_cols = [3*s+2 for s in sites]
    dbv_data = [1 for s in sites] # dv_s = 0

    rows += dbv_rows
    columns += dbv_cols
    data += dbv_data

    ###########################################################################
    #                top boundary: 0 <= i <= Nx-1 and j = Ny-1                #
    ###########################################################################
    # We want the last 2 rows to be equal

    # list of the sites in the top row
    sites = [i + (Ny-1)*Nx for i in range(Nx)]

    # top_n
    dtn_rows = [[3*s, 3*s] for s in sites]
    dtn_cols = [[3*(s-Nx), 3*s] for s in sites]
    dtn_data = [[1, -1] for s in sites]

    rows += list(chain.from_iterable(dtn_rows))
    columns += list(chain.from_iterable(dtn_cols))
    data += list(chain.from_iterable(dtn_data))

    # top_p
    dtp_rows = [[3*s+1, 3*s+1] for s in sites]
    dtp_cols = [[3*(s-Nx)+1, 3*s+1] for s in sites]
    dtp_data = [[1, -1] for s in sites]

    rows += list(chain.from_iterable(dtp_rows))
    columns += list(chain.from_iterable(dtp_cols))
    data += list(chain.from_iterable(dtp_data))

    # top_v
    dtv_rows = [[3*s+2, 3*s+2] for s in sites]
    dtv_cols = [[3*(s-Nx)+2, 3*s+2] for s in sites]
    dtv_data = [[1, -1] for s in sites]

    rows += list(chain.from_iterable(dtv_rows))
    columns += list(chain.from_iterable(dtv_cols))
    data += list(chain.from_iterable(dtv_data))

    ###########################################################################
    #              bottom boundary: 0 <= i <= Nx-1 and j = 0                  #
    ###########################################################################
    # We want the first 2 rows to be equal

    # list of the sites in the bottom row
    sites = [i for i in range(Nx)]

    # bottom_n
    dbn_rows = [[3*s, 3*s] for s in sites]
    dbn_cols = [[3*s, 3*(s+Nx)] for s in sites]
    dbn_data = [[1, -1] for s in sites]

    rows += list(chain.from_iterable(dbn_rows))
    columns += list(chain.from_iterable(dbn_cols))
    data += list(chain.from_iterable(dbn_data))

    # bottom_p
    dbp_rows = [[3*s+1, 3*s+1] for s in sites]
    dbp_cols = [[3*s+1, 3*(s+Nx)+1] for s in sites]
    dbp_data = [[1, -1] for s in sites]

    rows += list(chain.from_iterable(dbp_rows))
    columns += list(chain.from_iterable(dbp_cols))
    data += list(chain.from_iterable(dbp_data))

    # bottom_v
    dbv_rows = [[3*s+2, 3*s+2] for s in sites]
    dbv_cols = [[3*s+2, 3*(s+Nx)+2] for s in sites]
    dbv_data = [[1, -1] for s in sites]

    rows += list(chain.from_iterable(dbv_rows))
    columns += list(chain.from_iterable(dbv_cols))
    data += list(chain.from_iterable(dbv_data))

    J = coo_matrix((data, (rows, columns)), shape=(3*Nx*Ny, 3*Nx*Ny), dtype=np.float64)
    return J
