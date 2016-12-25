import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from itertools import chain

from .observables import get_n, get_p
from .defects  import defectsF, defectsJ
# remember that efn and efp are zero at equilibrium

def getFandJ_eq(sys, v, use_mumps):
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

        global rows, columns, data

        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        # compute differences of potentials
        dv_sm1, dv_sp1, dv_smN, dv_spN, dv_smNN, dv_spNN = 0, 0, 0, 0, 0, 0
        dv = -drho_dv[sites]

        if dx.all() != 0:
            dv_sp1 = -1 / (dx * dxbar)
            dv += -dv_sp1
            rows += sites.tolist()
            columns += (sites+1).tolist()
            data += dv_sp1.tolist()
        if dxm1.all() != 0:
            dv_sm1 = -1 / (dxm1 * dxbar)
            dv += -dv_sm1
            rows += sites.tolist()
            columns += (sites-1).tolist()
            data += dv_sm1.tolist()
        if dy.all() != 0:
            dv_spN = -1 / (dy * dybar)
            dv += -dv_spN
            rows += sites.tolist()
            columns += (sites+Nx).tolist()
            data += dv_spN.tolist()
        if dym1.all() != 0:
            dv_smN = 1 / (dym1 * dybar)
            dv += -dv_smN
            rows += sites.tolist()
            columns += (sites-Nx).tolist()
            data += dv_smN.tolist()
        if dz.all() != 0:
            dv_spNN = -1 / (dz * dzbar)
            dv += -dv_spNN
            rows += sites.tolist()
            columns += (sites+Nx*Ny).tolist()
            data += dv_spNN.tolist()
        if dzm1.all() != 0:
            dv_smNN = -1 / (dzm1 * dzbar)
            dv += -dv_smNN
            rows += sites.tolist()
            columns += (sites-Nx*Ny).tolist()
            data += dv_smNN.tolist()

        rows += sites.tolist()
        columns += sites.tolist()
        data += dv.tolist()

    ###########################################################################
    #                     For all sites in the system                         #
    ###########################################################################
    # carrier densities
    n = sys.Nc * np.exp(-sys.bl + v)
    p = sys.Nv * np.exp(-Eg + bl - v)

    # bulk charges
    rho = sys.rho - n + p
    drho_dv = -n - p

    # charge defects
    if len(sys.extra_charge_sites) != 0:
        defectsF(sys, n, p, rho)
        defectsJ(sys, n, p, drho_dv)

    # charge devided by epsilon
    rho = rho / sys.epsilon
    drho_dv = drho_dv / sys.epsilon

    # reshape the array as array[y-indices, x-indices]
    _sites = np.arange(Nx*Ny*Nz, dtype=int).reshape(Nz, Ny, Nx)
     
    ###########################################################################
    #     inside the system: 0 < i < Nx-1,  0 < j < Ny-1, 0 < k < Nz-1        #
    ###########################################################################
    # We compute fn, fp, fv derivatives. Those functions are only defined on the
    # inner part of the system. All the edges containing boundary conditions.

    # list of the sites inside the system
    sites = _sites[1:Nz-1, 1:Ny-1, 1:Nx-1].flatten()

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

    vec[sites] = poisson(v, dxm1, dx, dym1, dy, dzm1, dz, sites)
    poisson_derivs(v, dxm1, dx, dym1, dy, dzm1, dz, sites)


    ###########################################################################
    #       left boundary: i = 0, 0 <= j <= Ny-1, 0 <= k <= Nz-1              #
    ###########################################################################
    # list of the sites on the left side
    sites = _sites[:, :, 0].flatten()

    # update vector
    av_rows = sites
    vec[av_rows] = 0 # to ensure Dirichlet BCs

    # update Jacobian
    dav_rows = sites
    dav_cols = sites
    dav_data = [1 for s in sites] # dv_s = 0

    rows += dav_rows.tolist()
    columns += dav_cols.tolist()
    data += dav_data


    ###########################################################################
    #       right boundary: i = Nx-1, 0 <= j <= Ny-1, 0 <= k <= Nz-1          #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[:, :, Nx-1].flatten()

    # update vector
    bv_rows = sites
    vec[bv_rows] = 0 # to ensure Dirichlet BCs

    # update Jacobian
    dbv_rows = sites
    dbv_cols = sites
    dbv_data = [1 for s in sites] # dv_s = 0

    rows += dbv_rows.tolist()
    columns += dbv_cols.tolist()
    data += dbv_data


    ###########################################################################
    #           boundary: 0 < i < Nx-1, j = Ny-1, 0 < k < Nz-1                #
    ###########################################################################

    # list of the sites in the top row
    sites = _sites[1:Nz-1, Ny-1, 1:Nx-1].flatten()

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
    

    ###########################################################################
    #          bottom boundary: 0 < i < Nx-1, j = 0, 0 < k < Nz-1             #
    ###########################################################################
    # list of the sites in the bottom row
    sites = _sites[1:Nz-1, 0, 1:Nx-1].flatten()

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

    ###########################################################################
    #             boundary: 0 < i < Nx-1, 0 < j < Ny-1,  k = Nz-1             #
    ###########################################################################
    sites = _sites[Nz-1, 1:Ny-1, 1:Nx-1].flatten()

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

    ###########################################################################
    #             boundary: 0 < i < Nx-1, 0 < j < Ny-1,  k = 0                #
    ###########################################################################
    sites = _sites[0, 1:Ny-1, 1:Nx-1].flatten()

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

    ###########################################################################
    #                   boundary: 0 < i < Nx-1, j = 0,  k = 0                 #
    ###########################################################################
    sites = _sites[0, 0, 1:Nx-1].flatten()

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

    ###########################################################################
    #                   boundary: 0 < i < Nx-1, j = 0,  k = Nz-1              #
    ###########################################################################
    sites = _sites[Nz-1, 0, 1:Nx-1].flatten()

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

    ###########################################################################
    #                   boundary: 0 < i < Nx-1, j = Ny-1,  k = 0              #
    ###########################################################################
    sites = _sites[0, Ny-1, 1:Nx-1].flatten()

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

    ###########################################################################
    #                boundary: 0 < i < Nx-1, j = Ny-1,  k = Nz-1              #
    ###########################################################################
    sites = _sites[Nz-1, Ny-1, 1:Nx-1].flatten()

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

    # create the sparse matrix
    if use_mumps:
        J = coo_matrix((data, (rows, columns)), shape=(Nx*Ny*Nz, Nx*Ny*Nz), dtype=np.float64)
    else:
        J = csr_matrix((data, (rows, columns)), shape=(Nx*Ny*Nz, Nx*Ny*Nz), dtype=np.float64)

    return vec, J
