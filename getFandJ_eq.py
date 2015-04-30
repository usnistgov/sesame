import numpy as np
from scipy.sparse import coo_matrix
from itertools import chain

from sesame.observables import get_n, get_p
# remember that efn and efp are zero at equilibrium

def getFandJ_eq(v, params):
    bl = params.bl
    eg = params.eg
    xpts = params.xpts
    ypts = params.ypts
    rho = params.rho
    nC = params.nC
    nV = params.nV
    NGB = params.NGB
    nGB = params.nGB
    pGB = params.pGB

    Nx = xpts.shape[0]
    Ny = ypts.shape[0]

    delta_x = xpts[1:] - xpts[:-1]
    delta_y = ypts[1:] - ypts[:-1]

    # expand dx and dy in y and x respectively
    dx = np.tile(delta_x, (Ny, 1)).T
    dy = np.tile(delta_y, (Nx, 1))

    # reshape the vectors to conform to [x,y] coordinates
    v_xy = v.reshape(Ny, Nx).T
    rho_xy = rho.reshape(Ny, Nx).T
    NGB_xy = NGB.reshape(Ny, Nx).T

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

    ###########################################################################
    #       inside the system: 0 < i < Nx-1 and 0 < j < Ny-1                  #
    ###########################################################################
    # We compute fn, fp, fv derivatives. Those functions are only defined on the
    # inner part of the system. All the edges containing boundary conditions.

    # list of the sites inside the system
    sites = [i + j*Nx for j in range(1,Ny-1) for i in range(1,Nx-1)]

    # dxbar and dybar
    dxbar = (dx[1:,1:-1] + dx[:-1,1:-1]) / 2.
    dybar = (dy[1:-1,1:] + dy[1:-1,:-1]) / 2.

    # carrier densities
    n_xy = get_n(0, v_xy[1:-1,1:-1], params)
    p_xy = get_p(0, v_xy[1:-1,1:-1], params)

    # GB charge density
    fGB = (n_xy + pGB) / (n_xy + p_xy + nGB + pGB)
    rhoGB = NGB_xy[1:-1,1:-1] / 2. * (1 - 2*fGB)

    # GB charge density derivatives
    drhoGB_dv = -NGB_xy[1:-1,1:-1] * (n_xy*(n_xy+p_xy+nGB+pGB)-(n_xy+pGB)*(n_xy-p_xy))\
                                   / (n_xy+p_xy+nGB+pGB)**2

    #--------------------------------------------------------------------------
    #------------------------------ fv ----------------------------------------
    #--------------------------------------------------------------------------
    fv = ((v_xy[1:-1, 1:-1] - v_xy[:-2, 1:-1]) / dx[:-1,1:-1]\
         -(v_xy[2:, 1:-1] - v_xy[1:-1, 1:-1]) / dx[1:,1:-1]) / dxbar\
         +((v_xy[1:-1, 1:-1] - v_xy[1:-1, :-2]) / dy[1:-1,:-1]\
         -(v_xy[1:-1, 2:] - v_xy[1:-1, 1:-1]) / dy[1:-1,1:]) / dybar\
         -(rho_xy[1:-1, 1:-1] + rhoGB + p_xy - n_xy)

    # reshape the arrays as 1D arrays
    fv = (fv.T).reshape((Nx-2)*(Ny-2))

    # update the vector rows for the inner part of the system
    fv_rows = [s for s in sites]
    vec[fv_rows] = fv

    #--------------------------------------------------------------------------
    #-------------------------- fv derivatives --------------------------------
    #--------------------------------------------------------------------------
    # compute the derivatives
    dvmN = -1./(dy[1:-1,:-1] * dybar)
    dvm1 = -1./(dx[:-1,1:-1] * dxbar)
    dv = 2./(dx[1:,1:-1] * dx[:-1,1:-1]) + 2./(dy[1:-1,1:] * dy[1:-1,:-1])\
          + p_xy + n_xy - drhoGB_dv
    dvp1 = -1./(dx[1:,1:-1] * dxbar)
    dvpN = -1./(dy[1:-1,1:] * dybar)

    # reshape the derivatives as 1D arrays
    dvmN = (dvmN.T).reshape((Nx-2)*(Ny-2)) 
    dvm1 = (dvm1.T).reshape((Nx-2)*(Ny-2))
    dv   = (dv.T).reshape((Nx-2)*(Ny-2))
    dvp1 = (dvp1.T).reshape((Nx-2)*(Ny-2))
    dvpN = (dvpN.T).reshape((Nx-2)*(Ny-2))

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = [5*[s] for s in sites]

    dfv_cols = [[s-Nx, s-1, s, s+1, s+Nx] for s in sites]

    dfv_data = zip(dvmN, dvm1, dv, dvp1, dvpN)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))

    ###########################################################################
    #                  left boundary: i = 0 and 0 < j < Ny-1                  #
    ###########################################################################
    # list of the sites on the left side
    sites = [j*Nx for j in range(Ny)]

    # update vector
    av_rows = [s for s in sites]
    vec[av_rows] = 0 # to ensure Dirichlet BCs

    # update Jacobian
    dav_rows = [s for s in sites]
    dav_cols = [s for s in sites]
    dav_data = [1 for s in sites] # dv_s = 0

    rows += dav_rows
    columns += dav_cols
    data += dav_data

    ###########################################################################
    #                right boundary: i = Nx-1 and 0 < j < Ny-1                #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + j*Nx for j in range(Ny)]

    # update vector
    bv_rows = [s for s in sites]
    vec[bv_rows] = 0 # to ensure Dirichlet BCs

    # update Jacobian
    dbv_rows = [s for s in sites]
    dbv_cols = [s for s in sites]
    dbv_data = [1 for s in sites] # dv_s = 0

    rows += dbv_rows
    columns += dbv_cols
    data += dbv_data

    ###########################################################################
    #                top boundary: 0 <= i <= Nx-1 and j = Ny-1                #
    ###########################################################################
    # We want the last 2 rows to be equal

    # list of the sites in the top row
    sites = [i + (Ny-1)*Nx for i in range(1,Nx-1)]

    # update vector
    # we want zeros in the vector, so nothing to do

    # update Jacobian
    dtv_rows = [[s, s] for s in sites]
    dtv_cols = [[s, s-Nx] for s in sites]
    dtv_data = [[1, -1] for s in sites]

    rows += list(chain.from_iterable(dtv_rows))
    columns += list(chain.from_iterable(dtv_cols))
    data += list(chain.from_iterable(dtv_data))

    ###########################################################################
    #              bottom boundary: 0 <= i <= Nx-1 and j = 0                  #
    ###########################################################################
    # We want the first 2 rows to be equal

    # list of the sites in the bottom row
    sites = [i for i in range(1,Nx-1)]

    # update vector
    # we want zeros in the vector, so nothing to do

    # update Jacobian
    dbv_rows = [[s, s] for s in sites]
    dbv_cols = [[s, s+Nx] for s in sites]
    dbv_data = [[1, -1] for s in sites]

    rows += list(chain.from_iterable(dbv_rows))
    columns += list(chain.from_iterable(dbv_cols))
    data += list(chain.from_iterable(dbv_data))


    J = coo_matrix((data, (rows, columns)), shape=(Nx*Ny, Nx*Ny), dtype=np.float64)
    return vec, J
