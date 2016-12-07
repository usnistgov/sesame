import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from itertools import chain

from .observables import get_n, get_p
# remember that efn and efp are zero at equilibrium

def getFandJ_eq(sys, v, with_mumps):
    Nx, Ny, Nz = sys.xpts.shape[0], sys.ypts.shape[0], sys.zpts.shape[0]

    # lists of rows, columns and data that will create the sparse Jacobian
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

    def laplacian(vsmNN, vsmN, vsm1, vs, vsp1, vspN, vspNN, dxm1, dx, dym1, dy, 
                  dzm1, dz, dxbar, dybar, dzbar):
        res = ((vs - vsm1) / dxm1 - (vsp1 - vs) / dx) / dxbar\
            + ((vs - vsmN) / dym1 - (vspN - vs) / dy) / dybar\
            + ((vs - vsmNN) / dzm1 - (vspNN - vs) / dz) / dzbar 
        return res

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
        for idx, matches in enumerate(sys.extra_charge_sites):
            nextra = sys.nextra[idx, matches]
            pextra = sys.pextra[idx, matches]
            _n = n[matches]
            _p = p[matches]

            Se = sys.Seextra[idx, matches]
            Sh = sys.Shextra[idx, matches]
            f = (Se*_n + Sh*pextra) / (Se*(_n+nextra) + Sh*(_p+pextra))
            rho[matches] += sys.Nextra[idx, matches] / 2. * (1 - 2*f)

            drho_dv[matches] += - sys.Nextra[idx, matches]\
                                * (_n*(_n+_p+nextra+pextra)-(_n+pextra)*(_n-_p))\
                                / (_n+_p+nextra+pextra)**2

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
    fv = ((v[sites]-v[sites-1]) / dxm1 - (v[sites+1]-v[sites]) / dx) / dxbar\
       + ((v[sites]-v[sites-Nx]) / dym1 - (v[sites+Nx]-v[sites]) / dy) / dybar\
       + ((v[sites]-v[sites-Nx*Ny]) / dzm1 - (v[sites+Nx*Ny]-v[sites])\
       / dz) / dzbar - rho[sites]

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #-------------------------- fv derivatives --------------------------------
    dvmNN = -1./(dzm1 * dzbar)
    dvmN = -1./(dym1 * dybar)
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) + 2./(dy * dym1) + 2./(dz * dzm1) - drho_dv[sites]
    dvp1 = -1./(dx * dxbar)
    dvpN = -1./(dy * dybar)
    dvpNN = -1./(dz * dzbar)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = [7*[s] for s in sites]

    dfv_cols = [[s-Nx*Ny, s-Nx, s-1, s, s+1, s+Nx, s+Nx*Ny] for s in sites]

    dfv_data = zip(dvmNN, dvmN, dvm1, dv, dvp1, dvpN, dvpNN)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))


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
    #           boundary: 0 < i < Nx-1, j = Ny-1, 0 < k < Nz-1               #
    ###########################################################################
    # We want periodic boundary conditions. This means that we can apply Poisson
    # equation assuming that the potential outside the system is the same as the
    # one on the opposite edge.

    # list of the sites in the top row
    sites = [i + (Ny-1)*Nx + k*Nx*Ny for k in range(1,Nz-1) for i in\
                                                       range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Nz-2)
    dy = np.repeat((sys.dy[0] + sys.dy[-1])/2, (Nx-2)*(Nz-2))
    dz = np.repeat(sys.dz[:-1], Nx-2)
    dxm1 = np.tile(sys.dx[:-1], Nz-2)
    dym1 = np.repeat(sys.dy[-1], (Nx-2)*(Nz-2))
    dzm1 = np.zeros(Nz-1)
    dzm1 = np.repeat(sys.dz[:-1], Nx-2)
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    #---------------------------------- fv -------------------------------------
    vsmNN = v[sites-Nx*Ny]
    vsmN = v[sites-Nx]
    vsm1 = v[sites-1]
    vs = v[sites]
    vsp1 = v[sites+1]
    vspN = v[sites - Nx*(Ny-1)] # BC
    vspNN = v[sites + Nx*Ny]

    fv = laplacian(vsmNN, vsmN, vsm1, vs, vsp1, vspN, vspNN, dxm1, dx, \
                   dym1, dy, dzm1, dz, dxbar, dybar, dzbar) - rho[sites]

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #-------------------------- fv derivatives --------------------------------
    dvmNN = -1./(dzm1 * dzbar)
    dvmN = -1./(dym1 * dybar)
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) + (1/dy + 1/dym1)/dybar + (1/dz + 1/dzm1)/dzbar - drho_dv[sites]
    dvp1 = -1./(dx * dxbar)
    dvpN = -1./(dy * dybar) # BC
    dvpNN = -1./(dz * dzbar)

    # update the sparse matrix row and columns
    dfv_rows = [7*[s] for s in sites]
    dfv_cols = [[s-Nx*Ny, s-Nx, s-1, s, s+1, s-Nx*(Ny-1), s+Nx*Ny] for s in sites]
    dfv_data = zip(dvmNN, dvmN, dvm1, dv, dvp1, dvpN, dvpNN)


    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))


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
    dym1 = np.repeat((sys.dy[0] + sys.dy[-1])/2, (Nx-2)*(Nz-2))
    dzm1 = np.repeat(sys.dz[:-1], Nx-2)
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    #---------------------------------- fv -------------------------------------
    vsmNN = v[sites-Nx*Ny]
    vsmN = v[sites + Nx*(Ny-1)] # BC
    vsm1 = v[sites-1]
    vs = v[sites]
    vsp1 = v[sites+1]
    vspN = v[sites + Nx]
    vspNN = v[sites + Nx*Ny]

    fv = laplacian(vsmNN, vsmN, vsm1, vs, vsp1, vspN, vspNN, dxm1, dx, \
                   dym1, dy, dzm1, dz, dxbar, dybar, dzbar) - rho[sites]

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #-------------------------- fv derivatives --------------------------------
    dvmNN = -1./(dzm1 * dzbar)
    dvmN = -1./(dym1 * dybar) # BC
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) + (1/dy + 1/dym1)/dybar + (1/dz + 1/dzm1)/dzbar - drho_dv[sites]
    dvp1 = -1./(dx * dxbar)
    dvpN = -1./(dy * dybar)
    dvpNN = -1./(dz * dzbar)

    # update the sparse matrix row and columns
    dfv_rows = [7*[s] for s in sites]
    dfv_cols = [[s-Nx*Ny, s+Nx*(Ny-1), s-1, s, s+1, s+Nx, s+Nx*Ny] for s in sites]
    dfv_data = zip(dvmNN, dvmN, dvm1, dv, dvp1, dvpN, dvpNN)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))

    ###########################################################################
    #             boundary: 0 < i < Nx-1, 0 < j < Ny-1,  k = Nz-1             #
    ###########################################################################
    sites = [i + j*Nx + (Nz-1)*Nx*Ny for j in range(1,Ny-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny-2)
    dy = np.repeat(sys.dy[1:], Nx-2)
    dz = np.repeat((sys.dz[-1] + sys.dz[0])/2., (Nx-2)*(Ny-2))
    dxm1 = np.tile(sys.dx[:-1], Ny-2)
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dzm1 = np.repeat(sys.dz[-1], (Nx-2)*(Ny-2))
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    #---------------------------------- fv -------------------------------------
    vsmNN = v[sites - Nx*Ny]
    vsmN = v[sites - Nx]
    vsm1 = v[sites - 1]
    vs = v[sites]
    vsp1 = v[sites + 1]
    vspN = v[sites + Nx]
    vspNN = v[sites - Nx*Ny*(Nz-1)] # BC

    fv = laplacian(vsmNN, vsmN, vsm1, vs, vsp1, vspN, vspNN, dxm1, dx, \
                   dym1, dy, dzm1, dz, dxbar, dybar, dzbar) - rho[sites]

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #-------------------------- fv derivatives --------------------------------
    dvmNN = -1./(dzm1 * dzbar)
    dvmN = -1./(dym1 * dybar)
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) + (1/dy + 1/dym1)/dybar + (1/dz + 1/dzm1)/dzbar - drho_dv[sites]
    dvp1 = -1./(dx * dxbar)
    dvpN = -1./(dy * dybar)
    dvpNN = -1./(dz * dzbar)

    # update the sparse matrix row and columns
    dfv_rows = [7*[s] for s in sites]
    dfv_cols = [[s-Nx*Ny*(Nz-1), s-Nx*Ny, s-Nx, s-1, s, s+1,s+Nx] for s in sites]
    dfv_data = zip(dvpNN, dvmNN, dvmN, dvm1, dv, dvp1, dvpN)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))


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
    dzm1 = np.repeat((sys.dz[-1] + sys.dz[0])/2., (Nx-2)*(Ny-2))
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    #---------------------------------- fv -------------------------------------
    vsmNN = v[sites + Nx*Ny*(Nz-1)] # BC
    vsmN = v[sites - Nx]
    vsm1 = v[sites - 1]
    vs = v[sites]
    vsp1 = v[sites + 1]
    vspN = v[sites + Nx]
    vspNN = v[sites + Nx*Ny]

    fv = laplacian(vsmNN, vsmN, vsm1, vs, vsp1, vspN, vspNN, dxm1, dx, \
                   dym1, dy, dzm1, dz, dxbar, dybar, dzbar) - rho[sites]

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #-------------------------- fv derivatives --------------------------------
    dvmNN = -1./(dzm1 * dzbar)
    dvmN = -1./(dym1 * dybar)
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) + (1/dy + 1/dym1)/dybar + (1/dz + 1/dzm1)/dzbar - drho_dv[sites]
    dvp1 = -1./(dx * dxbar)
    dvpN = -1./(dy * dybar)
    dvpNN = -1./(dz * dzbar)

    # update the sparse matrix row and columns
    dfv_rows = [7*[s] for s in sites]
    dfv_cols = [[s-Nx, s-1, s, s+1, s+Nx, s+Nx*Ny, s+Nx*Ny*(Nz-1)] for s in sites]
    dfv_data = zip(dvmN, dvm1, dv, dvp1, dvpN, dvpNN, dvmNN)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))

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
    dym1 = np.repeat((sys.dy[0] + sys.dy[-1])/2., Nx-2)
    dzm1 = np.repeat((sys.dz[0] + sys.dz[-1])/2., Nx-2)
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    #---------------------------------- fv -------------------------------------
    vsmNN = v[sites + Nx*Ny*(Nz-1)] # BC
    vsmN = v[sites + Nx*(Ny-1)] # BC
    vsm1 = v[sites - 1]
    vs = v[sites]
    vsp1 = v[sites + 1]
    vspN = v[sites + Nx]
    vspNN = v[sites + Nx*Ny]

    fv = laplacian(vsmNN, vsmN, vsm1, vs, vsp1, vspN, vspNN, dxm1, dx, \
                   dym1, dy, dzm1, dz, dxbar, dybar, dzbar) - rho[sites]

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #-------------------------- fv derivatives --------------------------------
    dvmNN = -1./(dzm1 * dzbar)
    dvmN = -1./(dym1 * dybar)
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) + (1/dy + 1/dym1)/dybar + (1/dz + 1/dzm1)/dzbar - drho_dv[sites]
    dvp1 = -1./(dx * dxbar)
    dvpN = -1./(dy * dybar)
    dvpNN = -1./(dz * dzbar)

    # update the sparse matrix row and columns
    dfv_rows = [7*[s] for s in sites]
    dfv_cols = [[s-1, s, s+1, s+Nx, s+Nx*(Ny-1), s+Nx*Ny, s+Nx*Ny*(Nz-1)] for s in sites]
    dfv_data = zip(dvm1, dv, dvp1, dvpN, dvmN, dvpNN, dvmNN)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))


    ###########################################################################
    #                   boundary: 0 < i < Nx-1, j = 0,  k = Nz-1              #
    ###########################################################################
    sites = [i + (Nz-1)*Nx*Ny for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = sys.dx[1:]
    dy = np.repeat(sys.dy[0], Nx-2)
    dz = np.repeat((sys.dz[0] + sys.dz[-1])/2., Nx-2)
    dxm1 = sys.dx[:-1]
    dym1 = np.repeat((sys.dy[0] + sys.dy[-1])/2., Nx-2)
    dzm1 = np.repeat(sys.dz[-1], Nx-2)
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    #---------------------------------- fv -------------------------------------
    vsmNN = v[sites - Nx*Ny]
    vsmN = v[sites + Nx*(Ny-1)] # BC
    vsm1 = v[sites - 1]
    vs = v[sites]
    vsp1 = v[sites + 1]
    vspN = v[sites + Nx]
    vspNN = v[sites - Nx*Ny*(Nz-1)] # BC

    fv = laplacian(vsmNN, vsmN, vsm1, vs, vsp1, vspN, vspNN, dxm1, dx, \
                   dym1, dy, dzm1, dz, dxbar, dybar, dzbar) - rho[sites]

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #-------------------------- fv derivatives --------------------------------
    dvmNN = -1./(dzm1 * dzbar)
    dvmN = -1./(dym1 * dybar)
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) + (1/dy + 1/dym1)/dybar + (1/dz + 1/dzm1)/dzbar - drho_dv[sites]
    dvp1 = -1./(dx * dxbar)
    dvpN = -1./(dy * dybar)
    dvpNN = -1./(dz * dzbar)

    # update the sparse matrix row and columns
    dfv_rows = [7*[s] for s in sites]
    dfv_cols = [[s-Nx*Ny*(Nz-1), s-Nx*Ny, s-1, s, s+1, s+Nx, s+Nx*(Ny-1)] for s in sites]
    dfv_data = zip(dvpNN, dvmNN, dvm1, dv, dvp1, dvpN, dvmN)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))


    ###########################################################################
    #                   boundary: 0 < i < Nx-1, j = Ny-1,  k = 0              #
    ###########################################################################
    sites = [i + (Ny-1)*Nx for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = sys.dx[1:]
    dy = np.repeat((sys.dy[0] + sys.dy[-1])/2., Nx-2)
    dz = np.repeat(sys.dz[0], Nx-2)
    dxm1 = sys.dx[:-1]
    dym1 = np.repeat(sys.dy[-1], Nx-2)
    dzm1 = np.repeat((sys.dz[0] + sys.dz[-1])/2., Nx-2)
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    #---------------------------------- fv -------------------------------------
    vsmNN = v[sites + Nx*Ny*(Nz-1)] # BC
    vsmN = v[sites - Nx]
    vsm1 = v[sites - 1]
    vs = v[sites]
    vsp1 = v[sites + 1]
    vspN = v[sites - Nx*(Ny-1)] # BC
    vspNN = v[sites + Nx*Ny]

    fv = laplacian(vsmNN, vsmN, vsm1, vs, vsp1, vspN, vspNN, dxm1, dx, \
                   dym1, dy, dzm1, dz, dxbar, dybar, dzbar) - rho[sites]

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #-------------------------- fv derivatives --------------------------------
    dvmNN = -1./(dzm1 * dzbar)
    dvmN = -1./(dym1 * dybar)
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) + (1/dy + 1/dym1)/dybar + (1/dz + 1/dzm1)/dzbar - drho_dv[sites]
    dvp1 = -1./(dx * dxbar)
    dvpN = -1./(dy * dybar)
    dvpNN = -1./(dz * dzbar)

    # update the sparse matrix row and columns
    dfv_rows = [7*[s] for s in sites]
    dfv_cols = [[s-Nx*(Ny-1), s-Nx, s-1, s, s+1, s+Nx*Ny, s+Nx*Ny*(Nz-1)] for s in sites]
    dfv_data = zip(dvpN, dvmN, dvm1, dv, dvp1, dvpNN, dvmNN)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))


    ###########################################################################
    #                boundary: 0 < i < Nx-1, j = Ny-1,  k = Nz-1              #
    ###########################################################################
    sites = [i + (Ny-1)*Nx + (Nz-1)*Nx*Ny for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = sys.dx[1:]
    dy = np.repeat((sys.dy[0] + sys.dy[-1])/2., Nx-2)
    dz = np.repeat((sys.dz[0] + sys.dz[-1])/2., Nx-2)
    dxm1 = sys.dx[:-1]
    dym1 = np.repeat(sys.dy[-1], Nx-2)
    dzm1 = np.repeat(sys.dz[-1], Nx-2)
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.
    dzbar = (dz + dzm1) / 2.

    #---------------------------------- fv -------------------------------------
    vsmNN = v[sites - Nx*Ny]
    vsmN = v[sites - Nx]
    vsm1 = v[sites - 1]
    vs = v[sites]
    vsp1 = v[sites + 1]
    vspN = v[sites - Nx*(Ny-1)] #BC
    vspNN = v[sites - Nx*Ny*(Nz-1)] # BC

    fv = laplacian(vsmNN, vsmN, vsm1, vs, vsp1, vspN, vspNN, dxm1, dx, \
                   dym1, dy, dzm1, dz, dxbar, dybar, dzbar) - rho[sites]

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #-------------------------- fv derivatives --------------------------------
    dvmNN = -1./(dzm1 * dzbar)
    dvmN = -1./(dym1 * dybar)
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) + (1/dy + 1/dym1)/dybar + (1/dz + 1/dzm1)/dzbar - drho_dv[sites]
    dvp1 = -1./(dx * dxbar)
    dvpN = -1./(dy * dybar)
    dvpNN = -1./(dz * dzbar)

    # update the sparse matrix row and columns
    dfv_rows = [7*[s] for s in sites]
    dfv_cols = [[s-Nx*Ny*(Nz-1), s-Nx*Ny, s-Nx*(Ny-1), s-Nx, s-1, s, s+1] for s in sites]
    dfv_data = zip(dvpNN, dvmNN, dvpN, dvmN, dvm1, dv, dvp1)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))


    if with_mumps:
        J = coo_matrix((data, (rows, columns)), shape=(Nx*Ny*Nz, Nx*Ny*Nz), dtype=np.float64)
    else:
        J = csc_matrix((data, (rows, columns)), shape=(Nx*Ny*Nz, Nx*Ny*Nz), dtype=np.float64)
    return vec, J
