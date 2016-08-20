import numpy as np
from scipy.sparse import coo_matrix
from itertools import chain

from sesame.observables import get_n, get_p
# remember that efn and efp are zero at equilibrium

def getFandJ_eq(sys, v):
    Nx = sys.xpts.shape[0]
    dx = sys.dx
    
    # lists of rows, columns and data that will create the sparse Jacobian
    rows = []
    columns = []
    data = []

    # right hand side vector
    vec = np.zeros((Nx,))

    ###########################################################################
    #                     organization of the Jacobian matrix                 #
    ###########################################################################
    # A site with coordinates (i) corresponds to a site number s as follows:
    # i = s
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

    ###########################################################################
    #                 inside the system: 0 < i < Nx-1                         #
    ###########################################################################
    # We compute fn, fp, fv derivatives. Those functions are only defined on the
    # inner part of the system. All the edges containing boundary conditions.

    # list of the sites inside the system
    sites = [i for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # dxbar
    dxbar = (dx[sites] + dx[sites-1]) / 2.

    # carrier densities
    n = get_n(sys, 0*v, v, sites)
    p = get_p(sys, 0*v, v, sites)

    # bulk charges
    rho = sys.rho[sites] - n + p
    drho_dv = -n - p

    # extra charge density
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

    #--------------------------------------------------------------------------
    #------------------------------ fv ----------------------------------------
    #--------------------------------------------------------------------------
    fv = ((v[sites]-v[sites-1]) / dx[sites-1] - (v[sites+1]-v[sites]) / dx[sites]) / dxbar\
       - rho

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #--------------------------------------------------------------------------
    #-------------------------- fv derivatives --------------------------------
    #--------------------------------------------------------------------------
    # compute the derivatives
    dvm1 = -1./(dx[sites-1] * dxbar)
    dv = 2./(dx[sites] * dx[sites-1]) - drho_dv
    dvp1 = -1./(dx[sites] * dxbar)

    # update the sparse matrix row and columns for the inner part of the system
    dfv_rows = [3*[s] for s in sites]

    dfv_cols = [[s-1, s, s+1] for s in sites]

    dfv_data = zip(dvm1, dv, dvp1)

    rows += list(chain.from_iterable(dfv_rows))
    columns += list(chain.from_iterable(dfv_cols))
    data += list(chain.from_iterable(dfv_data))


    ###########################################################################
    #                          left boundary: i = 0                           #
    ###########################################################################
    # update vector
    vec[0] = 0 # to ensure Dirichlet BCs

    # update Jacobian
    dav_rows = [0]
    dav_cols = [0]
    dav_data = [1] # dv_s = 0

    rows += dav_rows
    columns += dav_cols
    data += dav_data


    ###########################################################################
    #                         right boundary: i = Nx-1                        #
    ###########################################################################
    # update vector
    vec[Nx-1] = 0 # to ensure Dirichlet BCs

    # update Jacobian
    dbv_rows = [Nx-1]
    dbv_cols = [Nx-1]
    dbv_data = [1] # dv_s = 0

    rows += dbv_rows
    columns += dbv_cols
    data += dbv_data

    J = coo_matrix((data, (rows, columns)), shape=(Nx, Nx), dtype=np.float64)
    return vec, J
