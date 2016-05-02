import numpy as np
from scipy.sparse import coo_matrix
from itertools import chain

from sesame.observables2 import get_n, get_p
# remember that efn and efp are zero at equilibrium

def getFandJ_eq(sys, v):
    Nx, Ny, Nz = sys.xpts.shape[0], sys.ypts.shape[0], sys.zpts.shape[0]

    # lists of rows, columns and data that will create the sparse Jacobian
    global rows, columns, data
    rows = []
    columns = []
    data = []

    # right hand side vector
    vec = np.zeros((Nx*Ny*Nz,))

    ###########################################################################
    #                     organization of the Jacobian matrix                 #
    ###########################################################################
    # A site with coordinates (i,j,k) corresponds to a site number s as follows:
    # k = s//(Nx*Ny)
    # j = s - s//Nx
    # i = s - j*Nx - k*Nx*Ny
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

    def poisson(v, dxm1, dx, dym1, dy, dzm1, dz, sites):
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        dv_sm1, dv_sp1, dv_smN, dv_spN, dv_smNN, dv_spNN = 0, 0, 0, 0, 0, 0
        v_s = v[sites]
        if dx.all() != 0:
            dv_sp1 = (v[sites+1] - v_s) / dx
        if dxm1.all() != 0:
            dv_sm1 = (v_s - v[sites-1]) / dxm1
        if dy.all() != 0:
            dv_spN = (v[sites+Nx] - v_s) / dy
        if dym1.all() != 0:
            dv_smN = (v_s - v[sites-Nx]) / dym1
        if dz.all() != 0:
            dv_spNN = (v[sites+Nx*Ny] - v_s) / dz
        if dzm1.all() != 0:
            dv_smNN = (v_s - v[sites-Nx*Ny]) / dzm1

        fv = (dv_sm1 - dv_sp1) / dxbar + (dv_smN - dv_spN) / dybar\
           + (dv_smNN - dv_spNN) / dzbar - rho[sites]

        return fv

    def poisson_derivs(v, dxm1, dx, dym1, dy, dzm1, dz, sites):
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        # compute differences of potentials
        dv_sm1, dv_sp1, dv_smN, dv_spN, dv_smNN, dv_spNN = [0], [0], [0], [0], [0], [0]
        dv = -drho_dv[sites]

        if dx.all() != 0:
            dv_sp1 = -1 / (dx * dxbar)
            dv += -dv_sp1
        if dxm1.all() != 0:
            dv_sm1 = -1 / (dxm1 * dxbar)
            dv += -dv_sm1
        if dy.all() != 0:
            dv_spN = -1 / (dy * dybar)
            dv += -dv_spN
        if dym1.all() != 0:
            dv_smN = 1 / (dym1 * dybar)
            dv += -dv_smN
        if dz.all() != 0:
            dv_spNN = -1 / (dz * dzbar)
            dv += -dv_spNN
        if dzm1.all() != 0:
            dv_smNN = -1 / (dzm1 * dzbar)
            dv += -dv_smNN

        dfv_rows = [7*[s] for s in sites]

        dfv_cols = [[s-Nx*Ny, s-Nx, s-1, s, s+1, s+Nx, s+Nx*Ny] for s in sites]

        dfv_data = zip(dv_smNN, dv_smN, dv_sm1, dv, dv_sp1, dv_spN, dv_spNN)

        global rows, columns, data
        rows += list(chain.from_iterable(dfv_rows))
        columns += list(chain.from_iterable(dfv_cols))
        data += list(chain.from_iterable(dfv_data))
        print(len(rows), len(columns), len(data))



    # def laplacian(vsmNN, vsmN, vsm1, vs, vsp1, vspN, vspNN, dxm1, dx, dym1, dy, 
    #               dzm1, dz, dxbar, dybar, dzbar):
    #     res = ((vs - vsm1) / dxm1 - (vsp1 - vs) / dx) / dxbar\
    #         + ((vs - vsmN) / dym1 - (vspN - vs) / dy) / dybar\
    #         + ((vs - vsmNN) / dzm1 - (vspNN - vs) / dz) / dzbar 
    #     return res

    ###########################################################################
    #                     For all sites in the system                         #
    ###########################################################################
    sites = [i + j*Nx + k*Nx*Ny for k in range(Nz) for j in range(Ny) for i in range(Nx)]

    # carrier densities
    n = get_n(sys, 0*v, v, sites)
    p = get_p(sys, 0*v, v, sites)

    # bulk charges
    rho = sys.rho[sites] - n + p
    drho_dv = -n - p
    
    # extra charge density
    if hasattr(sys, 'Nextra'): 
        nextra = sys.nextra[sites]
        pextra = sys.pextra[sites]

        f = (n + pextra) / (n + p + nextra + pextra)
        rho += sys.Nextra[sites] / 2. * (1 - 2*f)

        # GB charge density derivatives
        drho_dv += -sys.Nextra[sites] * (n*(n+p+nextra+pextra)-(n+pextra)*(n-p))\
                                       / (n+p+nextra+pextra)**2

    # charge is divided by epsilon (Poisson equation)
    rho = rho / sys.epsilon[sites]
    drho_dv = drho_dv / sys.epsilon[sites]

     
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
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    #------------------------------ fv ----------------------------------------
    # fv = laplacian(v[sites-Nx*Ny], v[sites-Nx], v[sites-1], v[sites], v[sites+1],
    # v[sites+Nx], v[sites+Nx*Ny], dxm1, dx, dym1, dy, dzm1, dz, dxbar, dybar,\
    # dzbar) - rho[sites]

    vec[sites] = poisson(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    poisson_derivs(v, dxm1, dx, dym1, dy, dzm1, dz, sites)

    # #-------------------------- fv derivatives --------------------------------
    # dvmNN = -1./(dzm1 * dzbar)
    # dvmN = -1./(dym1 * dybar)
    # dvm1 = -1./(dxm1 * dxbar)
    # dv = 2./(dx * dxm1) + 2./(dy * dym1) + 2./(dz * dzm1) - drho_dv[sites]
    # dvp1 = -1./(dx * dxbar)
    # dvpN = -1./(dy * dybar)
    # dvpNN = -1./(dz * dzbar)
    #
    # # update the sparse matrix row and columns for the inner part of the system
    # dfv_rows = [7*[s] for s in sites]
    #
    # dfv_cols = [[s-Nx*Ny, s-Nx, s-1, s, s+1, s+Nx, s+Nx*Ny] for s in sites]
    #
    # dfv_data = zip(dvmNN, dvmN, dvm1, dv, dvp1, dvpN, dvpNN)
    #
    # rows += list(chain.from_iterable(dfv_rows))
    # columns += list(chain.from_iterable(dfv_cols))
    # data += list(chain.from_iterable(dfv_data))


    ###########################################################################
    #       left boundary: i = 0, 0 <= j <= Ny-1, 0 <= k <= Nz-1              #
    ###########################################################################
    # list of the sites on the left side
    sites = [j*Nx + k*Nx*Ny for k in range(Nz) for j in range(Ny)]

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
    #       right boundary: i = Nx-1, 0 <= j <= Ny-1, 0 <= k <= Nz-1          #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + j*Nx + k*Nx*Ny for k in range(Nz) for j in range(Ny)]

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
    #           boundary: 0 < i < Nx-1, j = Ny-1, 0 < k < Nz-1                #
    ###########################################################################

    # list of the sites in the top row
    sites = [i + (Ny-1)*Nx + k*Nx*Ny for k in range(1,Nz-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Nz-2)
    dy = np.array([0])
    dz = np.repeat(sys.dz[:-1], Nx-2)
    dxm1 = np.tile(sys.dx[:-1], Nz-2)
    dym1 = np.repeat(sys.dy[-1], (Nx-2)*(Nz-2))
    dzm1 = np.zeros(Nz-1)
    dzm1 = np.repeat(sys.dz[:-1], Nx-2)
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    vec[sites] = poisson(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    poisson_derivs(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    #---------------------------------- fv -------------------------------------
    # vsmNN = v[sites-Nx*Ny]
    # vsmN = v[sites-Nx]
    # vsm1 = v[sites-1]
    # vs = v[sites]
    # vsp1 = v[sites+1]
    # vspNN = v[sites + Nx*Ny]
    #
    # fv = ((v_s - v_sm1) / dxm1 - (v_sp1 - v_s) / dx) / dxbar\
    #    + ((v_s - v_smN) / dym1) / dybar\
    #    + ((v_s - v_smNN) / dzm1 - (v_spNN - v_s) / dz) / dzbar\
    #    - rho[sites]
    #
    #
    # # update the vector rows for the inner part of the system
    # vec[sites] = fv
    #
    # #-------------------------- fv derivatives --------------------------------
    # dvmNN = -1./(dzm1 * dzbar)
    # dvmN = -1./(dym1 * dybar)
    # dvm1 = -1./(dxm1 * dxbar)
    # dv = 2./(dx * dxm1) + 1/(dym1*dybar) + 2/(dzm1*dzm1) - drho_dv[sites]
    # dvp1 = -1./(dx * dxbar)
    # dvpNN = -1./(dz * dzbar)

    # update the sparse matrix row and columns
    # dfv_rows = [6*[s] for s in sites]
    # dfv_cols = [[s-Nx*Ny, s-Nx, s-1, s, s+1, s+Nx*Ny] for s in sites]
    # dfv_data = zip(dvmNN, dvmN, dvm1, dv, dvp1, dvpNN)
    #
    #
    # rows += list(chain.from_iterable(dfv_rows))
    # columns += list(chain.from_iterable(dfv_cols))
    # data += list(chain.from_iterable(dfv_data))


    ###########################################################################
    #          bottom boundary: 0 < i < Nx-1, j = 0, 0 < k < Nz-1             #
    ###########################################################################
    # list of the sites in the bottom row
    sites = [i + k*Nx*Ny for k in range(1,Nz-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Nz-2)
    dy = np.repeat(sys.dy[0], (Nx-2)*(Nz-2))
    dz = np.repeat(sys.dz[1:], Nx-2)
    dxm1 = np.tile(sys.dx[:-1], Nz-2)
    dym1 = np.array([0])
    dzm1 = np.repeat(sys.dz[:-1], Nx-2)
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    vec[sites] = poisson(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    poisson_derivs(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    #---------------------------------- fv -------------------------------------
    # vsmNN = v[sites-Nx*Ny]
    # vsm1 = v[sites-1]
    # vs = v[sites]
    # vsp1 = v[sites+1]
    # vspN = v[sites + Nx]
    # vspNN = v[sites + Nx*Ny]
    #
    # fv = ((vs - vsm1) / dxm1 - (vsp1 - vs) / dx) / dxbar\
    #    + (-(vspN - vs) / dy) / dybar\
    #    + ((vs - vsmNN) / dzm1 - (vspNN - vs) / dz) / dzbar\
    #    - rho[sites]
    # # update the vector rows for the inner part of the system
    # vec[sites] = fv
    #
    # #-------------------------- fv derivatives --------------------------------
    # dvmNN = -1./(dzm1 * dzbar)
    # dvm1 = -1./(dxm1 * dxbar)
    # dv = 2./(dx * dxm1) + 1/(dy*dybar) + 2/(dz*dzm1) - drho_dv[sites]
    # dvp1 = -1./(dx * dxbar)
    # dvpN = -1./(dy * dybar)
    # dvpNN = -1./(dz * dzbar)
    #
    # # update the sparse matrix row and columns
    # dfv_rows = [6*[s] for s in sites]
    # dfv_cols = [[s-Nx*Ny, s-1, s, s+1, s+Nx, s+Nx*Ny] for s in sites]
    # dfv_data = zip(dvmNN, dvm1, dv, dvp1, dvpN, dvpNN)
    #
    # rows += list(chain.from_iterable(dfv_rows))
    # columns += list(chain.from_iterable(dfv_cols))
    # data += list(chain.from_iterable(dfv_data))

    ###########################################################################
    #             boundary: 0 < i < Nx-1, 0 < j < Ny-1,  k = Nz-1             #
    ###########################################################################
    sites = [i + j*Nx + (Nz-1)*Nx*Ny for j in range(1,Ny-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny-2)
    dy = np.repeat(sys.dy[1:], Nx-2)
    dz = np.array([0])
    dxm1 = np.tile(sys.dx[:-1], Ny-2)
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dzm1 = np.repeat(sys.dz[-1], (Nx-2)*(Ny-2))
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    vec[sites] = poisson(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    poisson_derivs(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    #---------------------------------- fv -------------------------------------
    # vsmNN = v[sites - Nx*Ny]
    # vsmN = v[sites - Nx]
    # vsm1 = v[sites - 1]
    # vs = v[sites]
    # vsp1 = v[sites + 1]
    # vspN = v[sites + Nx]
    #
    # fv = ((vs - vsm1) / dxm1 - (vsp1 - vs) / dx) / dxbar\
    #    + ((vs - vsmN) / dym1 - (vspN - vs) / dy) / dybar\
    #    + ((vs - vsmNN) / dzm1) / dzbar - rho[sites]
    #
    # # update the vector rows for the inner part of the system
    # vec[sites] = fv
    #
    # #-------------------------- fv derivatives --------------------------------
    # dvmNN = -1./(dzm1 * dzbar)
    # dvmN = -1./(dym1 * dybar)
    # dvm1 = -1./(dxm1 * dxbar)
    # dv = 2./(dx * dxm1) + 2/(dy*dym1) + (1/dzm1)/dzbar - drho_dv[sites]
    # dvp1 = -1./(dx * dxbar)
    # dvpN = -1./(dy * dybar)
    #
    # # update the sparse matrix row and columns
    # dfv_rows = [6*[s] for s in sites]
    # dfv_cols = [[s-Nx*Ny, s-Nx, s-1, s, s+1,s+Nx] for s in sites]
    # dfv_data = zip(dvmNN, dvmN, dvm1, dv, dvp1, dvpN)
    #
    # rows += list(chain.from_iterable(dfv_rows))
    # columns += list(chain.from_iterable(dfv_cols))
    # data += list(chain.from_iterable(dfv_data))


    ###########################################################################
    #             boundary: 0 < i < Nx-1, 0 < j < Ny-1,  k = 0                #
    ###########################################################################
    sites = [i + j*Nx for j in range(1,Ny-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny-2)
    dy = np.repeat(sys.dy[1:], Nx-2)
    dz = np.repeat(sys.dz[0], (Nx-2)*(Ny-2))
    dxm1 = np.tile(sys.dx[:-1], Ny-2)
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dzm1 = np.array([0])
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    vec[sites] = poisson(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    poisson_derivs(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    #---------------------------------- fv -------------------------------------
    # vsmN = v[sites - Nx]
    # vsm1 = v[sites - 1]
    # vs = v[sites]
    # vsp1 = v[sites + 1]
    # vspN = v[sites + Nx]
    # vspNN = v[sites + Nx*Ny]
    #
    # fv = ((vs - vsm1) / dxm1 - (vsp1 - vs) / dx) / dxbar\
    #    + ((vs - vsmN) / dym1 - (vspN - vs) / dy) / dybar\
    #    - ((vspNN - vs) / dz) / dzbar - rho[sites]
    #
    # # update the vector rows for the inner part of the system
    # vec[sites] = fv
    #
    # #-------------------------- fv derivatives --------------------------------
    # dvmN = -1./(dym1 * dybar)
    # dvm1 = -1./(dxm1 * dxbar)
    # dv = 2./(dx * dxm1) + 2/(dy*dym1) + (1/dz)/dzbar - drho_dv[sites]
    # dvp1 = -1./(dx * dxbar)
    # dvpN = -1./(dy * dybar)
    # dvpNN = -1./(dz * dzbar)
    #
    # # update the sparse matrix row and columns
    # dfv_rows = [6*[s] for s in sites]
    # dfv_cols = [[s-Nx, s-1, s, s+1, s+Nx, s+Nx*Ny] for s in sites]
    # dfv_data = zip(dvmN, dvm1, dv, dvp1, dvpN, dvpNN)
    #
    # rows += list(chain.from_iterable(dfv_rows))
    # columns += list(chain.from_iterable(dfv_cols))
    # data += list(chain.from_iterable(dfv_data))

    ###########################################################################
    #                   boundary: 0 < i < Nx-1, j = 0,  k = 0                 #
    ###########################################################################
    sites = [i for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = sys.dx[1:]
    dy = np.repeat(sys.dy[0], Nx-2)
    dz = np.repeat(sys.dz[0], Nx-2)
    dxm1 = sys.dx[:-1]
    dym1 = np.array([0])
    dzm1 = np.array([0])
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    vec[sites] = poisson(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    poisson_derivs(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    # #---------------------------------- fv -------------------------------------
    # vsm1 = v[sites - 1]
    # vs = v[sites]
    # vsp1 = v[sites + 1]
    # vspN = v[sites + Nx]
    # vspNN = v[sites + Nx*Ny]
    #
    # fv = ((vs - vsm1) / dxm1 - (vsp1 - vs) / dx) / dxbar\
    #    + (-(vspN - vs) / dy) / dybar\
    #    + (-(vspNN - vs) / dz) / dzbar\
    #    - rho[sites]
    #
    # # update the vector rows for the inner part of the system
    # vec[sites] = fv
    #
    # #-------------------------- fv derivatives --------------------------------
    # dvm1 = -1./(dxm1 * dxbar)
    # dv = 2./(dx * dxm1) + (1/dy)/dybar + (1/dz)/dzbar - drho_dv[sites]
    # dvp1 = -1./(dx * dxbar)
    # dvpN = -1./(dy * dybar)
    # dvpNN = -1./(dz * dzbar)
    #
    # # update the sparse matrix row and columns
    # dfv_rows = [5*[s] for s in sites]
    # dfv_cols = [[s-1, s, s+1, s+Nx, s+Nx*Ny] for s in sites]
    # dfv_data = zip(dvm1, dv, dvp1, dvpN, dvpNN)
    #
    # rows += list(chain.from_iterable(dfv_rows))
    # columns += list(chain.from_iterable(dfv_cols))
    # data += list(chain.from_iterable(dfv_data))


    ###########################################################################
    #                   boundary: 0 < i < Nx-1, j = 0,  k = Nz-1              #
    ###########################################################################
    sites = [i + (Nz-1)*Nx*Ny for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = sys.dx[1:]
    dy = np.repeat(sys.dy[0], Nx-2)
    dz = np.array([0])
    dxm1 = sys.dx[:-1]
    dym1 = np.array([0])
    dzm1 = np.repeat(sys.dz[-1], Nx-2)
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    vec[sites] = poisson(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    poisson_derivs(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    # #---------------------------------- fv -------------------------------------
    # vsmNN = v[sites - Nx*Ny]
    # vsm1 = v[sites - 1]
    # vs = v[sites]
    # vsp1 = v[sites + 1]
    # vspN = v[sites + Nx]
    #
    # fv = ((vs - vsm1) / dxm1 - (vsp1 - vs) / dx) / dxbar\
    #    + (- (vspN - vs) / dy) / dybar\
    #    + ((vs - vsmNN) / dzm1) / dzbar - rho[sites]
    #
    # # update the vector rows for the inner part of the system
    # vec[sites] = fv
    #
    # #-------------------------- fv derivatives --------------------------------
    # dvmNN = -1./(dzm1 * dzbar)
    # dvm1 = -1./(dxm1 * dxbar)
    # dv = 2./(dx * dxm1) + (1/dy)/dybar + (1/dzm1)/dzbar - drho_dv[sites]
    # dvp1 = -1./(dx * dxbar)
    # dvpN = -1./(dy * dybar)
    #
    # # update the sparse matrix row and columns
    # dfv_rows = [5*[s] for s in sites]
    # dfv_cols = [[s-Nx*Ny, s-1, s, s+1, s+Nx] for s in sites]
    # dfv_data = zip(dvmNN, dvm1, dv, dvp1, dvpN)
    #
    # rows += list(chain.from_iterable(dfv_rows))
    # columns += list(chain.from_iterable(dfv_cols))
    # data += list(chain.from_iterable(dfv_data))


    ###########################################################################
    #                   boundary: 0 < i < Nx-1, j = Ny-1,  k = 0              #
    ###########################################################################
    sites = [i + (Ny-1)*Nx for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = sys.dx[1:]
    dy = np.array([0])
    dz = np.repeat(sys.dz[0], Nx-2)
    dxm1 = sys.dx[:-1]
    dym1 = np.repeat(sys.dy[-1], Nx-2)
    dzm1 = np.array([0])
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    vec[sites] = poisson(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    poisson_derivs(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    # #---------------------------------- fv -------------------------------------
    # vsmN = v[sites - Nx]
    # vsm1 = v[sites - 1]
    # vs = v[sites]
    # vsp1 = v[sites + 1]
    # vspNN = v[sites + Nx*Ny]
    #
    # fv = ((vs - vsm1) / dxm1 - (vsp1 - vs) / dx) / dxbar\
    #    + ((vs - vsmN) / dym1) / dybar\
    #    + (- (vspNN - vs) / dz) / dzbar - rho[sites]
    #
    # # update the vector rows for the inner part of the system
    # vec[sites] = fv
    #
    # #-------------------------- fv derivatives --------------------------------
    # dvmN = -1./(dym1 * dybar)
    # dvm1 = -1./(dxm1 * dxbar)
    # dv = 2./(dx * dxm1) + (1/dym1)/dybar + (1/dz)/dzbar - drho_dv[sites]
    # dvp1 = -1./(dx * dxbar)
    # dvpNN = -1./(dz * dzbar)
    #
    # # update the sparse matrix row and columns
    # dfv_rows = [5*[s] for s in sites]
    # dfv_cols = [[s-Nx, s-1, s, s+1, s+Nx*Ny] for s in sites]
    # dfv_data = zip(dvmN, dvm1, dv, dvp1, dvpNN)
    #
    # rows += list(chain.from_iterable(dfv_rows))
    # columns += list(chain.from_iterable(dfv_cols))
    # data += list(chain.from_iterable(dfv_data))


    ###########################################################################
    #                boundary: 0 < i < Nx-1, j = Ny-1,  k = Nz-1              #
    ###########################################################################
    sites = [i + (Ny-1)*Nx + (Nz-1)*Nx*Ny for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = sys.dx[1:]
    dy = np.array([0])
    dz = np.array([0])
    dxm1 = sys.dx[:-1]
    dym1 = np.repeat(sys.dy[-1], Nx-2)
    dzm1 = np.repeat(sys.dz[-1], Nx-2)
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    vec[sites] = poisson(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    poisson_derivs(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    # #---------------------------------- fv -------------------------------------
    # vsmNN = v[sites - Nx*Ny]
    # vsmN = v[sites - Nx]
    # vsm1 = v[sites - 1]
    # vs = v[sites]
    # vsp1 = v[sites + 1]
    #
    # fv = ((vs - vsm1) / dxm1 - (vsp1 - vs) / dx) / dxbar\
    #    + ((vs - vsmN) / dym1) / dybar\
    #    + ((vs - vsmNN) / dzm1) / dzbar - rho[sites]
    #
    # # update the vector rows for the inner part of the system
    # vec[sites] = fv
    #
    # #-------------------------- fv derivatives --------------------------------
    # dvmNN = -1./(dzm1 * dzbar)
    # dvmN = -1./(dym1 * dybar)
    # dvm1 = -1./(dxm1 * dxbar)
    # dv = 2./(dx * dxm1) + (1/dym1)/dybar + (1/dzm1)/dzbar - drho_dv[sites]
    # dvp1 = -1./(dx * dxbar)
    #
    # # update the sparse matrix row and columns
    # dfv_rows = [5*[s] for s in sites]
    # dfv_cols = [[s-Nx*Ny, s-Nx, s-1, s, s+1] for s in sites]
    # dfv_data = zip(dvmNN, dvmN, dvm1, dv, dvp1)
    #
    # rows += list(chain.from_iterable(dfv_rows))
    # columns += list(chain.from_iterable(dfv_cols))
    # data += list(chain.from_iterable(dfv_data))


    J = coo_matrix((data, (rows, columns)), shape=(Nx*Ny*Nz, Nx*Ny*Nz), dtype=np.float64)

    return vec, J
