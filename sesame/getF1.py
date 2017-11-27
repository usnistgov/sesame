# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import numpy as np
from .observables import *

def getF(sys, v, efn, efp, veq):
    ###########################################################################
    #               organization of the right hand side vector                #
    ###########################################################################
    # A site with coordinates (i) corresponds to a site number s as follows:
    # i = s
    #
    # Rows for (efn_s, efp_s, v_s)
    # ----------------------------
    # fn_row = 3*s
    # fp_row = 3*s+1
    # fv_row = 3*s+2

    Nx = sys.xpts.shape[0]
    dx = sys.dx

    # right hand side vector
    vec = np.zeros((3*Nx,))

    ###########################################################################
    #                     For all sites in the system                         #
    ###########################################################################
    # carrier densities
    n = sys.Nc * np.exp(+sys.bl + efn + v)
    p = sys.Nv * np.exp(-sys.Eg - sys.bl + efp - v)

    # bulk charges
    rho = sys.rho - n + p

    # recombination rates
    r = get_bulk_rr(sys, n, p)

    # charge devided by epsilon
    rho = rho / sys.epsilon

    ###########################################################################
    #                   inside the system: 0 < i < Nx-1                       #
    ###########################################################################
    # We compute fn, fp, fv. Those functions are only defined on the
    # inner part of the system. All the edges containing boundary conditions.

    # list of the sites inside the system
    sites = np.arange(1,Nx-1, dtype=int)

    # dxbar
    dx = sys.dx[1:]
    dxm1 = sys.dx[:-1]
    dxbar = (dx + dxm1) / 2.

    # compute the currents
    jnx_s   = get_jn(sys, efn, v, sites, sites+1, dx)
    jnx_sm1 = get_jn(sys, efn, v, sites-1, sites, dxm1)

    jpx_s   = get_jp(sys, efp, v, sites, sites+1, dx)
    jpx_sm1 = get_jp(sys, efp, v, sites-1, sites, dxm1)

    #------------------------------ fn ----------------------------------------
    fn = (jnx_s - jnx_sm1) / dxbar + sys.g[sites] - r[sites]

    vec[3*sites] = fn

    #------------------------------ fp ----------------------------------------
    fp = (jpx_s - jpx_sm1) / dxbar + r[sites] - sys.g[sites]

    vec[3*sites+1] = fp

    #------------------------------ fv ----------------------------------------
    fv = ((v[sites]-v[sites-1]) / dxm1 - (v[sites+1]-v[sites]) / dx) / dxbar\
       - rho[sites]

    vec[3*sites+2] = fv

    ###########################################################################
    #                       left boundary: i = 0                              #
    ###########################################################################
    # compute the currents
    jnx = get_jn(sys, efn, v, 0, 1, sys.dx[0])
    jpx = get_jp(sys, efp, v, 0, 1, sys.dx[0])

    # compute an, ap, av
    n_eq = sys.Nc[0] * np.exp(+sys.bl[0] + veq[0])
    p_eq = sys.Nv[0] * np.exp(-sys.Eg[0] - sys.bl[0] - veq[0])
        
    an = jnx - sys.Scn[0] * (n[0] - n_eq)
    ap = jpx + sys.Scp[0] * (p[0] - p_eq)
    av = 0 # Dirichlet

    vec[0] = an
    vec[1] = ap
    vec[2] = av

    ###########################################################################
    #                         right boundary: i = Nx-1                        #
    ###########################################################################
    # dxbar
    dxbar = dx[-1]

    # compute the currents
    jnx_sm1 = get_jn(sys, efn, v, Nx-2, Nx-1, sys.dx[-1])
    jpx_sm1 = get_jp(sys, efp, v, Nx-2, Nx-1, sys.dx[-1])

    sites = Nx-1
    jnx_s = jnx_sm1 + dxbar * (r[sites] - sys.g[sites])
    jpx_s = jpx_sm1 + dxbar * (sys.g[sites] - r[sites])

    # b_n, b_p and b_v values
    n_eq = sys.Nc[-1] * np.exp(+sys.bl[-1] + veq[-1])
    p_eq = sys.Nv[-1] * np.exp(-sys.Eg[-1] - sys.bl[-1] - veq[-1])
        
    bn = jnx_s + sys.Scn[1] * (n[-1] - n_eq)
    bp = jpx_s - sys.Scp[1] * (p[-1] - p_eq)
    bv = 0

    vec[3*(Nx-1)] = bn
    vec[3*(Nx-1)+1] = bp
    vec[3*(Nx-1)+2] = bv      

    return vec
