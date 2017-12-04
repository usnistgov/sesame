# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import numpy as np
from itertools import chain

from .observables import get_n, get_p
# remember that efn and efp are zero at equilibrium

def getFandJ_eq(sys, v, contacts_bcs):
    Nx = sys.xpts.shape[0]
    
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
    #                     For all sites in the system                         #
    ###########################################################################
    # carrier densities
    n = sys.Nc * np.exp(+sys.bl + v)
    p = sys.Nv * np.exp(-sys.Eg - sys.bl - v)

    # bulk charges
    rho = sys.rho - n + p
    drho_dv = -n - p

    ###########################################################################
    #                 inside the system: 0 < i < Nx-1                         #
    ###########################################################################
    # We compute fn, fp, fv derivatives. Those functions are only defined on the
    # inner part of the system. All the edges containing boundary conditions.

    # list of the sites inside the system
    sites = np.array(range(1,Nx-1))

    # dxbar
    dx = sys.dx[1:]
    dxm1 = sys.dx[:-1]
    dxbar = (dx + dxm1) / 2.

    eps_av_p = 1 / 2. * (sys.epsilon[sites + 1] + sys.epsilon[sites])
    eps_av_m = 1 / 2. * (sys.epsilon[sites] + sys.epsilon[sites - 1])

    #--------------------------------------------------------------------------
    #------------------------------ fv ----------------------------------------
    #--------------------------------------------------------------------------
    fv = (eps_av_m * (v[sites] - v[sites - 1]) / dxm1 - eps_av_p * (v[sites + 1] - v[sites]) / dx) / dxbar \
         - rho[sites]

    # update the vector rows for the inner part of the system
    vec[sites] = fv

    #--------------------------------------------------------------------------
    #-------------------------- fv derivatives --------------------------------
    #--------------------------------------------------------------------------
    # compute the derivatives
    dvm1 = -eps_av_m * 1. / (dxm1 * dxbar)
    dv = eps_av_m / (dxm1 * dxbar) + eps_av_p / (dx * dxbar) - drho_dv[sites]
    dvp1 = -eps_av_p * 1. / (dx * dxbar)

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
    if contacts_bcs == "Neumann":
        # update vector with surface charges to zero => dV/dx = 0
        vec[0] = v[1]-v[0]
        # update Jacobian
        dv_s = -1
        dv_sp1 = 1
        dav_rows = [0, 0]
        dav_cols = [0, 1]
        dav_data = [dv_s, dv_sp1]

    if contacts_bcs == "Dirichlet":
        # update vector with zeros
        vec[0] = 0
        # update Jacobian
        dav_rows = [0]
        dav_cols = [0]
        dav_data = [1]

    rows += dav_rows
    columns += dav_cols
    data += dav_data

    ###########################################################################
    #                         right boundary: i = Nx-1                        #
    ###########################################################################
    if contacts_bcs == "Neumann":
        # update vector with surface charges to zero => dV/dx = 0
        vec[Nx-1] = v[-1] - v[-2]
        # update Jacobian
        dv_s = 1
        dv_sm1 = -1
        dbv_rows = [Nx-1, Nx-1]
        dbv_cols = [Nx-2, Nx-1]
        dbv_data = [dv_sm1, dv_s]

    if contacts_bcs == "Dirichlet":
        # update vector with zeros
        vec[Nx-1] = 0
        # update Jacobian
        dbv_rows = [Nx-1]
        dbv_cols = [Nx-1]
        dbv_data = [1]

    rows += dbv_rows
    columns += dbv_cols
    data += dbv_data

    return vec, rows, columns, data
