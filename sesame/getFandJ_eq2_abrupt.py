import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from itertools import chain

from sesame.observables import get_n, get_p
# remember that efn and efp are zero at equilibrium

def getFandJ_eq(sys, v, with_mumps):
    Nx, Ny = sys.xpts.shape[0], sys.ypts.shape[0]
    
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

    def laplacian(vsmN, vsm1, vs, vsp1, vspN, dxm1, dx, dym1, dy, dxbar, dybar):
        res = ((vs - vsm1) / dxm1 - (vsp1 - vs) / dx) / dxbar\
            + ((vs - vsmN) / dym1 - (vspN - vs) / dy) / dybar
        return res

    ###########################################################################
    #                     For all sites in the system                         #
    ###########################################################################
    sites = [i + j*Nx for j in range(Ny) for i in range(Nx)]

    # carrier densities
    n = get_n(sys, 0*v, v, sites)
    p = get_p(sys, 0*v, v, sites)

    # bulk charges
    rho = sys.rho[sites] - n + p
    drho_dv = -n - p

    if hasattr(sys, 'Nextra'): 
        # find sites containing extra charges
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
    #       inside the system: 0 < i < Nx-1 and 0 < j < Ny-1                  #
    ###########################################################################
    # We compute fn, fp, fv derivatives. Those functions are only defined on the
    # inner part of the system. All the edges containing boundary conditions.

    # list of the sites inside the system
    sites = [i + j*Nx for j in range(1,Ny-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny-2)
    dy = np.repeat(sys.dy[1:], Nx-2)
    dxm1 = np.tile(sys.dx[:-1], Ny-2)
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dxbar = (dx + dxm1) / 2.
    dybar = (dy + dym1) / 2.

    #------------------------------ fv ----------------------------------------
    fv = laplacian(v[sites-Nx], v[sites-1], v[sites], v[sites+1], v[sites+Nx],\
                   dxm1, dx, dym1, dy, dxbar, dybar) - rho[sites]

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #-------------------------- fv derivatives --------------------------------
    dvmN = -1./(dym1 * dybar)
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) + 2./(dy * dym1) - drho_dv[sites]
    dvp1 = -1./(dx * dxbar)
    dvpN = -1./(dy * dybar)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = [5*[s] for s in sites]

    dfv_cols = [[s-Nx, s-1, s, s+1, s+Nx] for s in sites]

    dfv_data = zip(dvmN, dvm1, dv, dvp1, dvpN)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))


    ###########################################################################
    #                   left contact: i = 0 and 0 <= j <= Ny-1                #
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
    #                 right contact: i = Nx-1 and 0 <= j <= Ny-1              #
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
    #                  boundary: 0 < i < Nx-1 and j = Ny-1                    #
    ###########################################################################

    # list of sites
    sites = np.asarray([i + (Ny-1)*Nx for i in range(1,Nx-1)])

    # lattice distances
    dx = sys.dx[1:]
    dxm1 = sys.dx[:-1]
    dy = 0
    dym1 = np.repeat(sys.dy[-1], Nx-2)
    dxbar = (dx + dxm1) / 2.
    dybar = dym1

    #---------------------------------- fv -------------------------------------
    fv = ((v[sites]-v[sites-1]) / dxm1 - (v[sites+1]-v[sites]) / dx) / dxbar\
       + ((v[sites]-v[sites-Nx]) / dym1) / dybar\
       - rho[sites]

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #-------------------------- fv derivatives --------------------------------
    dvmN = -1./(dym1 * dybar)
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) + 1/(dym1*dybar) - drho_dv[sites]
    dvp1 = -1./(dx * dxbar)

    # update the sparse matrix row and columns
    dfv_rows = [4*[s] for s in sites]
    dfv_cols = [[s-Nx, s-1, s, s+1] for s in sites]
    dfv_data = zip(dvmN, dvm1, dv, dvp1)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))

    ###########################################################################
    #                     boundary: 0 < i < Nx-1 and j = 0                    #
    ###########################################################################
    # list of sites
    sites = np.asarray([i for i in range(1,Nx-1)])

    # dxbar and dybar
    dx = sys.dx[1:]
    dxm1 = sys.dx[:-1]
    dy = np.repeat(sys.dy[0], Nx-2)
    dym1 = 0
    dxbar = (dx + dxm1) / 2.
    dybar = dy

    #---------------------------------- fv -------------------------------------
    fv = ((v[sites]-v[sites-1]) / dxm1 - (v[sites+1]-v[sites]) / dx) / dxbar\
       + (-(v[sites+Nx]-v[sites]) / dy) / dybar\
       - rho[sites]

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #-------------------------- fv derivatives --------------------------------
    dvm1 = -1./(dxm1 * dxbar)
    dv = 2./(dx * dxm1) + 1/(dy*dybar) - drho_dv[sites]
    dvp1 = -1./(dx * dxbar)
    dvpN = -1./(dy * dybar)

    # update the sparse matrix row and columns
    dfv_rows = [4*[s] for s in sites]
    dfv_cols = [[s-1, s, s+1, s+Nx] for s in sites]
    dfv_data = zip(dvm1, dv, dvp1, dvpN)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))


    if with_mumps == True:
        J = coo_matrix((data, (rows, columns)), shape=(Nx*Ny, Nx*Ny), dtype=np.float64)
    else:
        J = csr_matrix((data, (rows, columns)), shape=(Nx*Ny, Nx*Ny), dtype=np.float64)
    return vec, J
