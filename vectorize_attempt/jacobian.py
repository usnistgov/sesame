import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from itertools import chain

from sesame.jacobian_utils import *

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


    #--------------------------------------------------------------------------
    #--------------------- jv derivatives in the x-direction ------------------
    #--------------------------------------------------------------------------
    # jvn_s derivatives in the x-direction only for the inner part of the system
    djvnx_s_defn_s, djvnx_s_defn_sp1, \
    djvnx_s_dv_sm1, djvnx_s_dv_s, djvnx_s_dv_sp1, djvnx_s_dv_smN, djvnx_s_dv_spN = \
    mu[1:-1,1:-1] * get_jvn_s_derivs(efn_xy[1:-1,1:-1], efn_xy[2:,1:-1], \
    v_xy[:-2,1:-1], v_xy[1:-1,1:-1], v_xy[2:,1:-1], v_xy[1:-1,:-2],\
    v_xy[1:-1,2:], dx[1:,1:-1], params)

    # jvn_sm1 derivatives in the x-direction only for the inner part of the system
    djvnx_sm1_defn_sm1, djvnx_sm1_defn_s, \
    djvnx_sm1_dv_sm1, djvnx_sm1_dv_s, djvnx_sm1_dv_sp1, djvnx_sm1_dv_smN, djvnx_sm1_dv_spN = \
    mu[:-2,1:-1] * get_jvn_sm1_derivs(efn_xy[:-2,1:-1], efn_xy[1:-1,1:-1], \
    v_xy[:-2,1:-1], v_xy[1:-1,1:-1], v_xy[2:,1:-1], v_xy[1:-1,:-2],\
    v_xy[1:-1,2:], dx[:-1,1:-1], params)

    # jvp_s derivatives in the x-direction only for the inner part of the system
    djvpx_s_defp_s, djvpx_s_defp_sp1, \
    djvpx_s_dv_sm1, djvpx_s_dv_s, djvpx_s_dv_sp1, djvpx_s_dv_smN, djvpx_s_dv_spN = \
    mu[1:-1,1:-1] * get_jvp_s_derivs(efp_xy[1:-1,1:-1], efp_xy[2:,1:-1], \
    v_xy[:-2,1:-1], v_xy[1:-1,1:-1], v_xy[2:,1:-1], v_xy[1:-1,:-2],\
    v_xy[1:-1,2:], dx[1:,1:-1], params)

    # jvp_sm1 derivatives in the x-direction only for the inner part of the system
    djvpx_sm1_defp_sm1, djvpx_sm1_defp_s, \
    djvpx_sm1_dv_sm1, djvpx_sm1_dv_s, djvpx_sm1_dv_sp1, djvpx_sm1_dv_smN, djvpx_sm1_dv_spN = \
    mu[:-2,1:-1] * get_jvp_sm1_derivs(efp_xy[:-2,1:-1], efp_xy[1:-1,1:-1], \
    v_xy[:-2,1:-1], v_xy[1:-1,1:-1], v_xy[2:,1:-1], v_xy[1:-1,:-2],
    v_xy[1:-1,2:], dx[:-1,1:-1], params)


    #--------------------------------------------------------------------------
    #--------------------- jv derivatives in the y-direction ------------------
    #--------------------------------------------------------------------------
    # These are obtained by changing 1 by N and N by 1 in the definitions for the
    # x-direction: v_sm1 -> v_smN, v_spN -> v_sp1.
    # jvn_s derivatives in the y-direction only for the inner part of the system
    djvny_s_defn_s, djvny_s_defn_spN, \
    djvny_s_dv_sm1, djvny_s_dv_s, djvny_s_dv_sp1, djvny_s_dv_smN, djvny_s_dv_spN = \
    mu[1:-1,1:-1] * get_jvn_s_derivs(efn_xy[1:-1,1:-1], efn_xy[1:-1,2:], \
    v_xy[1:-1,:-2], v_xy[1:-1,1:-1], v_xy[1:-1,2:], v_xy[:-2,1:-1],\
    v_xy[2:,1:-1], dy[1:-1,1:], params)

    # jvn_smN derivatives in the y-direction only for the inner part of the system
    djvny_smN_defn_smN, djvny_smN_defn_s, \
    djvny_smN_dv_sm1, djvny_smN_dv_s, djvny_smN_dv_sp1, djvny_smN_dv_smN, djvny_smN_dv_spN = \
    mu[1:-1,:-2] * get_jvn_sm1_derivs(efn_xy[1:-1,:-2], efn_xy[1:-1,1:-1], \
    v_xy[1:-1,:-2], v_xy[1:-1,1:-1], v_xy[1:-1,2:], v_xy[:-2,1:-1],\
    v_xy[2:,1:-1], dy[1:-1,:-1], params)

    # jvp_s derivatives in the y-direction only for the inner part of the system
    djvpy_s_defp_s, djvpy_s_defp_spN, \
    djvpy_s_dv_sm1, djvpy_s_dv_s, djvpy_s_dv_sp1, djvpy_s_dv_smN, djvpy_s_dv_spN = \
    mu[1:-1,1:-1] * get_jvp_s_derivs(efp_xy[1:-1,1:-1], efp_xy[1:-1,2:], \
    v_xy[1:-1,:-2], v_xy[1:-1,1:-1], v_xy[1:-1,2:], v_xy[:-2,1:-1],\
    v_xy[2:,1:-1], dy[1:-1,1:], params)

    # jvp_smN derivatives in the y-direction only for the inner part of the system
    djvpy_smN_defp_smN, djvpy_smN_defp_s, \
    djvpy_smN_dv_sm1, djvpy_smN_dv_s, djvpy_smN_dv_sp1, djvpy_smN_dv_smN, djvpy_smN_dv_spN = \
    mu[1:-1,:-2] * get_jvp_sm1_derivs(efp_xy[1:-1,:-2], efp_xy[1:-1,1:-1], \
    v_xy[1:-1,:-2], v_xy[1:-1,1:-1], v_xy[1:-1,2:], v_xy[:-2,1:-1],\
    v_xy[2:,1:-1], dy[1:-1,:-1], params)


    #--------------------------------------------------------------------------
    #------------------------------ uv derivatives  ---------------------------
    #--------------------------------------------------------------------------
    # uvn derivatives
    duvn_defn_s, duvn_defp_s,\
    duvn_dv_smN, duvn_dv_sm1, duvn_dv_s, duvn_dv_sp1, duvn_dv_spN = \
    get_uvn_derivs(n_xy[1:-1,1:-1], p_xy[1:-1,1:-1], v_xy[1:-1,:-2],\
    v_xy[:-2,1:-1], v_xy[1:-1,1:-1], v_xy[2:,1:-1], v_xy[1:-1,2:], \
    g_xy[1:-1,1:-1], S_xy[1:-1,1:-1], SGB_xy[1:-1,1:-1], params)

    # uvp derivatives
    duvp_defn_s, duvp_defp_s, \
    duvp_dv_smN, duvp_dv_sm1, duvp_dv_s, duvp_dv_sp1, duvp_dv_spN = \
    get_uvp_derivs(n_xy[1:-1,1:-1], p_xy[1:-1,1:-1], v_xy[1:-1,:-2],
    v_xy[:-2,1:-1], v_xy[1:-1,1:-1], v_xy[2:,1:-1], v_xy[1:-1,2:], \
    g_xy[1:-1,1:-1], S_xy[1:-1,1:-1], SGB_xy[1:-1,1:-1], params)


    #--------------------------------------------------------------------------
    #------------------------------ GB charge density -------------------------
    #--------------------------------------------------------------------------
    drhoGB_dv = -NGB_xy * (n_xy*(n_xy+p_xy+nGB+pGB)-(n_xy+pGB)*(n_xy-p_xy))\
                        / (n_xy+p_xy+nGB+pGB)**2
    drhoGB_defn = -NGB_xy * (n_xy*(n_xy+p_xy+nGB+pGB)-(n_xy+pGB)*n_xy)\
                          / (n_xy+p_xy+nGB+pGB)**2
    drhoGB_defp = NGB_xy * (n_xy+pGB)*p_xy / (n_xy+p_xy+nGB+pGB)**2


    #--------------------------------------------------------------------------
    #---------------- fn derivatives inside the system ------------------------
    #--------------------------------------------------------------------------
    dxbar = (dx[1:,1:-1] + dx[:-1,1:-1]) / 2.
    dybar = (dy[1:-1,1:] + dy[1:-1,:-1]) / 2.
    # compute the derivatives
    dfn_defn_smN = - djvny_smN_defn_smN / dybar
    dfn_dv_smN = (djvnx_s_dv_smN - djvnx_sm1_dv_smN) / dxbar +\
                 (djvny_s_dv_smN - djvny_smN_dv_smN) / dybar +\
                 duvn_dv_smN

    dfn_defn_sm1 = - djvnx_sm1_dv_sm1 / dxbar
    dfn_dv_sm1 = (djvnx_s_dv_sm1 - djvnx_sm1_dv_sm1) / dxbar +\
                 (djvny_s_dv_sm1 - djvny_smN_dv_sm1) / dybar +\
                 duvn_dv_sm1

    dfn_defn_s = (djvnx_s_defn_s - djvnx_sm1_defn_s) / dxbar +\
                 (djvny_s_defn_s - djvny_smN_defn_s) / dybar +\
                 duvn_defn_s
    dfn_defp_s = duvn_defp_s
    dfn_dv_s = (djvnx_s_dv_s - djvnx_sm1_dv_s) / dxbar +\
               (djvny_s_dv_s - djvny_smN_dv_s) / dybar +\
               duvn_dv_s

    dfn_defn_sp1 = djvnx_s_defn_sp1 / dxbar
    dfn_dv_sp1 = (djvnx_s_dv_sp1 - djvnx_sm1_dv_sp1) / dxbar +\
                 (djvny_s_dv_sp1 - djvny_smN_dv_sp1) / dybar +\
                 duvn_dv_sp1

    dfn_defn_spN = djvny_s_defn_spN / dybar
    dfn_dv_spN = (djvnx_s_dv_spN - djvnx_sm1_dv_spN) / dxbar +\
                 (djvny_s_dv_spN - djvny_smN_dv_spN) / dybar +\
                 duvn_dv_spN

    # reshape the derivatives as 1D arrays
    dfn_defn_smN = (dfn_defn_smN.T).reshape((Nx-2)*(Ny-2))
    dfn_dv_smN   = (dfn_dv_smN.T).reshape((Nx-2)*(Ny-2))
    dfn_defn_sm1 = (dfn_defn_sm1.T).reshape((Nx-2)*(Ny-2))
    dfn_dv_sm1   = (dfn_dv_sm1.T).reshape((Nx-2)*(Ny-2))
    dfn_defn_s   = (dfn_defn_s.T).reshape((Nx-2)*(Ny-2))
    dfn_defp_s   = (dfn_defp_s.T).reshape((Nx-2)*(Ny-2))
    dfn_dv_s     = (dfn_dv_s.T).reshape((Nx-2)*(Ny-2))
    dfn_defn_sp1 = (dfn_defn_sp1.T).reshape((Nx-2)*(Ny-2))
    dfn_dv_sp1   = (dfn_dv_sp1.T).reshape((Nx-2)*(Ny-2))
    dfn_defn_spN = (dfn_defn_spN.T).reshape((Nx-2)*(Ny-2))
    dfn_dv_spN   = (dfn_dv_spN.T).reshape((Nx-2)*(Ny-2))

    # update the sparse matrix row and columns for the inner part of the system
    dfn_rows = [11*[3*s] for s in sites]

    dfn_cols = [[3*(s-Nx), 3*(s-Nx)+2, 3*(s-1), 3*(s-1)+2, 3*s, 3*s+1, 3*s+2,\
                3*(s+1), 3*(s+1)+2, 3*(s+Nx), 3*(s+Nx)+2] for s in sites]

    dfn_data = zip(dfn_defn_smN, dfn_dv_smN, dfn_defn_sm1, dfn_dv_sm1,\
                   dfn_defn_s, dfn_defp_s, dfn_dv_s, dfn_defn_sp1, dfn_dv_sp1,  
                   dfn_defn_spN, dfn_dv_spN)

    rows += list(chain.from_iterable(dfn_rows))
    columns += list(chain.from_iterable(dfn_cols))
    data += list(chain.from_iterable(dfn_data))

    #--------------------------------------------------------------------------
    #---------------- fp derivatives inside the system ------------------------
    #--------------------------------------------------------------------------
    # compute the derivatives
    dfp_defp_smN = - djvpy_smN_defp_smN / dybar
    dfp_dv_smN = (djvpx_s_dv_smN - djvpx_sm1_dv_smN) / dxbar +\
                 (djvpy_s_dv_smN - djvpy_smN_dv_smN) / dybar -\
                 duvp_dv_smN

    dfp_defp_sm1 = - djvpx_sm1_dv_sm1 / dxbar
    dfp_dv_sm1 = (djvpx_s_dv_sm1 - djvpx_sm1_dv_sm1) / dxbar +\
                 (djvpy_s_dv_sm1 - djvpy_smN_dv_sm1) / dybar -\
                 duvp_dv_sm1

    dfp_defn_s = - duvp_defn_s
    dfp_defp_s = (djvpx_s_defp_s - djvpx_sm1_defp_s) / dxbar +\
                 (djvpy_s_defp_s - djvpy_smN_defp_s) / dybar -\
                 duvp_defp_s
    dfp_dv_s = (djvpx_s_dv_s - djvpx_sm1_dv_s) / dxbar +\
               (djvpy_s_dv_s - djvpy_smN_dv_s) / dybar -\
               duvp_dv_s

    dfp_defp_sp1 = djvpx_s_defp_sp1 / dxbar
    dfp_dv_sp1 = (djvpx_s_dv_sp1 - djvpx_sm1_dv_sp1) / dxbar +\
                 (djvpy_s_dv_sp1 - djvpy_smN_dv_sp1) / dybar -\
                 duvp_dv_sp1

    dfp_defp_spN = djvpy_s_defp_spN / dybar
    dfp_dv_spN = (djvpx_s_dv_spN - djvpx_sm1_dv_spN) / dxbar +\
                 (djvpy_s_dv_spN - djvpy_smN_dv_spN) / dybar -\
                 duvp_dv_spN

    # reshape the derivatives as 1D arrays
    dfp_defp_smN = (dfp_defp_smN.T).reshape((Nx-2)*(Ny-2))
    dfp_dv_smN   = (dfp_dv_smN.T).reshape((Nx-2)*(Ny-2))
    dfp_defp_sm1 = (dfp_defp_sm1.T).reshape((Nx-2)*(Ny-2))
    dfp_dv_sm1   = (dfp_dv_sm1.T).reshape((Nx-2)*(Ny-2))
    dfp_defn_s   = (dfp_defn_s.T).reshape((Nx-2)*(Ny-2))
    dfp_defp_s   = (dfp_defp_s.T).reshape((Nx-2)*(Ny-2))
    dfp_dv_s     = (dfp_dv_s.T).reshape((Nx-2)*(Ny-2))
    dfp_defp_sp1 = (dfp_defp_sp1.T).reshape((Nx-2)*(Ny-2))
    dfp_dv_sp1   = (dfp_dv_sp1.T).reshape((Nx-2)*(Ny-2))
    dfp_defp_spN = (dfp_defp_spN.T).reshape((Nx-2)*(Ny-2))
    dfp_dv_spN   = (dfp_dv_spN.T).reshape((Nx-2)*(Ny-2))

    # update the sparse matrix row and columns for the inner part of the system
    dfp_rows = [11*[3*s+1] for s in sites]

    dfp_cols = [[3*(s-Nx)+1, 3*(s-Nx)+2, 3*(s-1)+1, 3*(s-1)+2, 3*s, 3*s+1,\
                 3*s+2, 3*(s+1)+1, 3*(s+1)+2, 3*(s+Nx)+1, 3*(s+Nx)+2] 
                 for s in sites]

    dfp_data = zip(dfp_defp_smN, dfp_dv_smN, dfp_defp_sm1, dfp_dv_sm1,\
                   dfp_defn_s, dfp_defp_s, dfp_dv_s, dfp_defp_sp1, dfp_dv_sp1,  
                   dfp_defp_spN, dfp_dv_spN)

    rows += list(chain.from_iterable(dfp_rows))
    columns += list(chain.from_iterable(dfp_cols))
    data += list(chain.from_iterable(dfp_data))

    #--------------------------------------------------------------------------
    #---------------- fv derivatives inside the system ------------------------
    #--------------------------------------------------------------------------
    # compute the derivatives
    dfv_dvmN = -1./(dy[1:-1,:-1] * dybar)
    dfv_dvm1 = -1./(dx[:-1,1:-1] * dxbar)
    dfv_dv = 2./(dx[1:,1:-1] * dx[:-1,1:-1]) + 2./(dy[1:-1,1:] * dy[1:-1,:-1])\
             + p_xy[1:-1,1:-1] + n_xy[1:-1,1:-1] - drhoGB_dv[1:-1,1:-1]
    dfv_defn = n_xy[1:-1,1:-1] - drhoGB_defn[1:-1,1:-1]
    dfv_defp = -p_xy[1:-1,1:-1] - drhoGB_defp[1:-1,1:-1]
    dfv_dvp1 = -1./(dx[1:,1:-1] * dxbar)
    dfv_dvpN = -1./(dy[1:-1,1:] * dybar)

    # reshape the derivatives as 1D arrays
    dfv_dvmN = (dfv_dvmN.T).reshape((Nx-2)*(Ny-2)) 
    dfv_dvm1 = (dfv_dvm1.T).reshape((Nx-2)*(Ny-2))
    dfv_dv   = (dfv_dv.T).reshape((Nx-2)*(Ny-2))
    dfv_defn = (dfv_defn.T).reshape((Nx-2)*(Ny-2))
    dfv_defp = (dfv_defp.T).reshape((Nx-2)*(Ny-2))
    dfv_dvp1 = (dfv_dvp1.T).reshape((Nx-2)*(Ny-2))
    dfv_dvpN = (dfv_dvpN.T).reshape((Nx-2)*(Ny-2))

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = [7*[3*s+2] for s in sites]

    dfv_cols = [[3*(s-Nx)+2, 3*(s-1)+2, 3*s, 3*s+1, 3*s+2, 3*(s+1)+2, 3*(s+Nx)+2]
                for s in sites]

    dfv_data = zip(dfv_dvmN, dfv_dvm1, dfv_defn, dfv_defp, dfv_dv, dfv_dvp1, dfv_dvpN)

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
    defn_s = -(v_xy[0,1:-1] - v_xy[1,1:-1]) * exp(efn_xy[0,1:-1]) / dx[0,1:-1]

    defn_sp1 = (v_xy[0,1:-1] - v_xy[1,1:-1]) * exp(efn_xy[1,1:-1]) / dx[0,1:-1]

    dv_s = (exp(efn_xy[1,1:-1]) - exp(efn_xy[0,1:-1])) / dx[0,1:-1]\
           - exp(-v_xy[0,1:-1]) * scn[0]*(n_xy[0,1:-1] - nD)\
           - (exp(-v_xy[1,1:-1]) - exp(-v_xy[0,1:-1])) * scn[0] * n_xy[0,1:-1]

    dv_sp1 = - (exp(efn_xy[1,1:-1]) - exp(efn_xy[0,1:-1])) / dx[0,1:-1]\
             + exp(-v_xy[1,1:-1]) * scn[0] * (n_xy[0,1:-1] - nD)

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
    defp_s = -(v_xy[0,1:-1] - v_xy[1,1:-1]) * exp(efp_xy[0,1:-1]) / dx[0,1:-1]

    defp_sp1 = (v_xy[0,1:-1] - v_xy[1,1:-1]) * exp(efp_xy[1,1:-1]) / dx[0,1:-1]

    dv_s = (exp(efp_xy[1,1:-1]) - exp(efp_xy[0,1:-1])) / dx[0,1:-1]\
           - exp(v_xy[0,1:-1]) * scp[0]*(p_xy[0,1:-1] - ni**2/nD)\
           + (exp(v_xy[1,1:-1]) - exp(v_xy[0,1:-1])) * scp[0] * p_xy[0,1:-1]

    dv_sp1 = - (exp(efp_xy[1,1:-1]) - exp(efp_xy[0,1:-1])) / dx[0,1:-1]\
             + exp(v_xy[1,1:-1]) * scp[0] * (p_xy[0,1:-1] - ni**2/nD)

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

    dxbar = dx[1,1:-1]
    dybar = (dy[-1,1:] + dy[-1,:-1]) / 2.

    # list of the sites on the right side
    sites = [Nx-1 + j*Nx for j in range(1,Ny-1)]

    #--------------------------------------------------------------------------
    #-------------------------- jvbn derivatives ------------------------------
    #--------------------------------------------------------------------------
    djvbnx_sm1_defn_sm1,  djvbnx_sm1_defn_s, djvbnx_sm1_dv_smN, djvbnx_sm1_dv_sm1,\
    djvbnx_sm1_dv_s, djvbnx_sm1_dv_spN = \
    mu[Nx-2, 1:-1] * get_jvbnx_sm1_derivs(efn_xy[Nx-2,1:-1], efn_xy[Nx-1,1:-1],\
    v_xy[Nx-2,1:-1], v_xy[Nx-1,1:-1], v_xy[Nx-1,:-2], v_xy[Nx-1,2:], dx[-2,1:-1], params)

    djvbny_s_defn_s,  djvbny_s_defn_spN, djvbny_s_dv_smN, djvbny_s_dv_sm1,\
    djvbny_s_dv_s, djvbny_s_dv_spN = \
    mu[Nx-1,1:-1] * get_jvbny_s_derivs(efn_xy[Nx-1,1:-1], efn_xy[Nx-1,2:],\
    v_xy[Nx-2,1:-1], v_xy[Nx-1,1:-1], v_xy[Nx-1,:-2], v_xy[Nx-1,2:], dy[-1,1:], params)

    djvbny_smN_defn_smN,  djvbny_smN_defn_s, djvbny_smN_dv_smN, djvbny_smN_dv_sm1,\
    djvbny_smN_dv_s, djvbny_smN_dv_spN = \
    mu[Nx-1,:-2] * get_jvbny_smN_derivs(efn_xy[Nx-1,:-2], efn_xy[Nx-1,1:-1],\
    v_xy[Nx-2,1:-1], v_xy[Nx-1,1:-1], v_xy[Nx-1,:-2], v_xy[Nx-1,2:], dy[-1,:-1], params)

    #--------------------------------------------------------------------------
    #-------------------------- uvbn derivatives ------------------------------
    #--------------------------------------------------------------------------
    duvbn_defn_s, duvbn_defp_s, \
    duvbn_dv_smN, duvbn_dv_sm1, duvbn_dv_s, duvbn_dv_spN =\
    get_uvbn_derivs(n_xy[Nx-1,1:-1], p_xy[Nx-1,1:-1], efn_xy[Nx-1,1:-1],\
    v_xy[Nx-2,1:-1], v_xy[Nx-1,1:-1], v_xy[Nx-1,:-2], v_xy[Nx-1,2:],\
    g_xy[-1,1:-1], S_xy[-1,1:-1], SGB_xy[-1,1:-1], params)


    #--------------------------------------------------------------------------
    #-------------------------- bn derivatives --------------------------------
    #--------------------------------------------------------------------------
    v_sm1 = v_xy[-2,1:-1] 
    v_smN = v_xy[-1,:-2] 
    v_s = v_xy[-1,1:-1]
    v_spN = v_xy[-1,2:] 

    defn_smN = dxbar / dybar * djvbny_smN_defn_smN
    dv_smN = djvbnx_sm1_dv_smN + dxbar * (-duvbn_dv_smN - (djvbny_s_dv_smN -\
                                          djvbny_smN_dv_smN) / dybar)\
            + scn[1] * (n_xy[-1,1:-1] - ni**2/nA) * exp(-v_smN) * \
            (exp(-v_s) - exp(-v_sm1)) * (exp(-v_spN) - exp(-v_s))

    defn_sm1 = djvbnx_sm1_defn_sm1
    dv_sm1 = djvbnx_sm1_dv_sm1 + dxbar * (-duvbn_dv_sm1 - (djvbny_s_dv_sm1 -\
                                          djvbny_smN_dv_sm1) / dybar)\
            + scn[1] * (n_xy[-1,1:-1] - ni**2/nA) * exp(-v_sm1) * \
            (exp(-v_spN) - exp(-v_s)) * (exp(-v_s) - exp(-v_smN))

    defn_s = djvbnx_sm1_defn_s + dxbar * (-duvbn_defn_s - (djvbny_s_defn_s -\
                                          djvbny_smN_defn_s) / dybar)\
            + scn[1] * n_xy[-1,1:-1] * (exp(-v_s) - exp(-v_sm1))\
            * (exp(-v_spN) - exp(-v_s)) * (exp(-v_s) - exp(-v_smN))
    defp_s = dxbar * (-duvbn_defp_s)
    dv_s = djvbnx_sm1_dv_s + dxbar * (-duvbn_dv_s - (djvbny_s_dv_s -\
                                          djvbny_smN_dv_s) / dybar)\
            + scn[1] * n_xy[-1,1:-1] * (exp(-v_s) - exp(-v_sm1))\
            * (exp(-v_spN) - exp(-v_s)) * (exp(-v_s) - exp(-v_smN))\
            + scn[1] * (n_xy[-1,1:-1] - ni**2/nA) * exp(-v_s) * \
            (-(exp(-v_spN) - exp(-v_s))*(exp(-v_s) - exp(-v_smN)) + (exp(-v_s) -
            exp(-v_sm1))*(exp(-v_s) - exp(-v_smN)) - (exp(-v_s) -
            exp(-v_sm1))*(exp(-v_spN) - exp(-v_s)))

    defn_spN = - dxbar / dybar * djvbny_s_defn_spN
    dv_spN = djvbnx_sm1_dv_spN + dxbar * (-duvbn_dv_spN - (djvbny_s_dv_spN -\
                                          djvbny_smN_dv_spN) / dybar)\
            - scn[1] * (n_xy[-1,1:-1] - ni**2/nA) * exp(-v_spN) * \
            (exp(-v_s) - exp(-v_sm1)) * (exp(-v_s) - exp(-v_smN))

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
    #-------------------------- jvbp derivatives ------------------------------
    #--------------------------------------------------------------------------
    djvbpx_sm1_defp_sm1,  djvbpx_sm1_defp_s, djvbpx_sm1_dv_smN, djvbpx_sm1_dv_sm1,\
    djvbpx_sm1_dv_s, djvbpx_sm1_dv_spN = \
    mu[Nx-2, 1:-1] * get_jvbpx_sm1_derivs(efp_xy[Nx-2,1:-1], efp_xy[Nx-1,1:-1],\
    v_xy[Nx-2,1:-1], v_xy[Nx-1,1:-1], v_xy[Nx-1,:-2], v_xy[Nx-1,2:], dx[-2,1:-1], params)

    djvbpy_s_defp_s,  djvbpy_s_defp_spN, djvbpy_s_dv_smN, djvbpy_s_dv_sm1,\
    djvbpy_s_dv_s, djvbpy_s_dv_spN = \
    mu[Nx-1,1:-1] * get_jvbpy_s_derivs(efp_xy[Nx-1,1:-1], efp_xy[Nx-1,2:],\
    v_xy[Nx-2,1:-1], v_xy[Nx-1,1:-1], v_xy[Nx-1,:-2], v_xy[Nx-1,2:], dy[-1,1:], params)

    djvbpy_smN_defp_smN,  djvbpy_smN_defp_s, djvbpy_smN_dv_smN, djvbpy_smN_dv_sm1,\
    djvbpy_smN_dv_s, djvbpy_smN_dv_spN = \
    mu[Nx-1,:-2] * get_jvbny_smN_derivs(efp_xy[Nx-1,:-2], efp_xy[Nx-1,1:-1],\
    v_xy[Nx-2,1:-1], v_xy[Nx-1,1:-1], v_xy[Nx-1,:-2], v_xy[Nx-1,2:], dy[-1,:-1], params)

    #--------------------------------------------------------------------------
    #-------------------------- uvbp derivatives ------------------------------
    #--------------------------------------------------------------------------
    duvbp_defn_s, duvbp_defp_s, \
    duvbp_dv_smN, duvbp_dv_sm1, duvbp_dv_s, duvbp_dv_spN =\
    get_uvbp_derivs(n_xy[Nx-1,1:-1], p_xy[Nx-1,1:-1], efp_xy[Nx-1,1:-1],\
    v_xy[Nx-2,1:-1], v_xy[Nx-1,1:-1], v_xy[Nx-1,:-2], v_xy[Nx-1,2:],\
    g_xy[-1,1:-1], S_xy[-1,1:-1], SGB_xy[-1,1:-1], params)


    #--------------------------------------------------------------------------
    #-------------------------- bp derivatives --------------------------------
    #--------------------------------------------------------------------------
    defp_smN = dxbar / dybar * djvbpy_smN_defp_smN
    dv_smN = djvbnx_sm1_dv_smN + dxbar * (duvbp_dv_smN - (djvbpy_s_dv_smN -\
                                          djvbpy_smN_dv_smN) / dybar)\
            - scp[1] * (p_xy[-1,1:-1] - nA) * exp(v_smN) * \
            (exp(v_s) - exp(v_sm1)) * (exp(v_spN) - exp(v_s))

    defp_sm1 = djvbpx_sm1_defp_sm1
    dv_sm1 = djvbpx_sm1_dv_sm1 + dxbar * (duvbp_dv_sm1 - (djvbpy_s_dv_sm1 -\
                                          djvbpy_smN_dv_sm1) / dybar)\
            - scp[1] * (p_xy[-1,1:-1] - nA) * exp(v_sm1) * \
            (exp(v_spN) - exp(v_s)) * (exp(v_s) - exp(v_smN))

    defn_s = dxbar * duvbp_defn_s
    defp_s = djvbpx_sm1_defp_s + dxbar * (duvbp_defp_s - (djvbpy_s_defp_s -\
                                          djvbpy_smN_defp_s) / dybar)\
            + scp[1] * p_xy[-1,1:-1] * (exp(v_s) - exp(v_sm1))\
            * (exp(v_spN) - exp(v_s)) * (exp(v_s) - exp(v_smN))
    dv_s = djvbpx_sm1_dv_s + dxbar * (duvbp_dv_s - (djvbpy_s_dv_s -\
                                          djvbpy_smN_dv_s) / dybar)\
            - scp[1] * p_xy[-1,1:-1] * (exp(v_s) - exp(v_sm1))\
            * (exp(v_spN) - exp(v_s)) * (exp(v_s) - exp(v_smN))\
            + scp[1] * (p_xy[-1,1:-1] - nA) * exp(v_s) * \
            ((exp(v_spN) - exp(v_s))*(exp(v_s) - exp(v_smN)) - (exp(v_s) -
            exp(v_sm1))*(exp(v_s) - exp(v_smN)) + (exp(v_s) -
            exp(v_sm1))*(exp(v_spN) - exp(v_s)))

    defp_spN = - dxbar / dybar * djvbpy_s_defp_spN
    dv_spN = djvbpx_sm1_dv_spN + dxbar * (duvbp_dv_spN - (djvbpy_s_dv_spN -\
                                          djvbpy_smN_dv_spN) / dybar)\
            + scp[1] * (p_xy[-1,1:-1] - nA) * exp(v_spN) * \
            (exp(v_s) - exp(v_sm1)) * (exp(v_s) - exp(v_smN))

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

    J = csr_matrix((data, (rows, columns)), shape=(3*Nx*Ny, 3*Nx*Ny), dtype=np.float64)
    return J
