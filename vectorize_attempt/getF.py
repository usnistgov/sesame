import numpy as np
from numpy import exp
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

from sesame.jacobian_utils import *

def getF(v, efn, efp, params):

    bl, eg, nC, nV, nA, nD, scn, scp, g, mu, tau, rho,\
    NGB, SGB, nGB, pGB,\
    n1, p1, ni, xpts, ypts = params

    Nx = xpts.shape[0]
    Ny = ypts.shape[0]

    delta_x = xpts[1:] - xpts[:-1]
    delta_y = ypts[1:] - ypts[:-1]

    dx = np.tile(delta_x, (Ny, 1)).T
    dy = np.tile(delta_y, (Nx, 1))

    dxbar = np.tile(delta_x[1:] + delta_x[:-1], (Ny, 1)).T / 2.
    dybar = np.tile(delta_y[1:] + delta_y[:-1], (Nx, 1)) / 2.

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

    vec = np.zeros((3*Nx*Ny,))

    ###########################################################################
    #       inside the system: 0 < i < Nx-1 and 0 < j < Ny-1                  #
    ###########################################################################
    # We compute fn, fp, fv derivatives. Those functions are only defined on the
    # inner part of the system. All the edges containing boundary conditions.

    # list of the sites inside the system
    sites = [i + j*Nx for i in range(1,Nx-1) for j in range(1,Ny-1)]

    #--------------------------------------------------------------------------
    #--------------------------- jv in the x-direction ------------------------
    #--------------------------------------------------------------------------
    # jvn_s in the x-direction only for the inner part of the system
    jvnx_s = mu[1:-1,1:-1] * get_jvn_s(efn_xy[1:-1,1:-1], efn_xy[2:,1:-1], \
    v_xy[:-2,1:-1], v_xy[1:-1,1:-1], v_xy[2:,1:-1], v_xy[1:-1,:-2],\
    v_xy[1:-1,2:], dx[1:,1:-1], params)

    # jvn_sm1 in the x-direction only for the inner part of the system
    jvnx_sm1 = mu[:-2,1:-1] * get_jvn_sm1(efn_xy[:-2,1:-1], efn_xy[1:-1,1:-1], \
    v_xy[:-2,1:-1], v_xy[1:-1,1:-1], v_xy[2:,1:-1], v_xy[1:-1,:-2],\
    v_xy[1:-1,2:], dx[:-1,1:-1], params)

    # jvp_s in the x-direction only for the inner part of the system
    jvpx_s = mu[1:-1,1:-1] * get_jvp_s(efp_xy[1:-1,1:-1], efp_xy[2:,1:-1], \
    v_xy[:-2,1:-1], v_xy[1:-1,1:-1], v_xy[2:,1:-1], v_xy[1:-1,:-2],\
    v_xy[1:-1,2:], dx[1:,1:-1], params)

    # jvp_sm1 in the x-direction only for the inner part of the system
    jvpx_sm1 = mu[:-2,1:-1] * get_jvp_sm1(efp_xy[:-2,1:-1], efp_xy[1:-1,1:-1], \
    v_xy[:-2,1:-1], v_xy[1:-1,1:-1], v_xy[2:,1:-1], v_xy[1:-1,:-2],
    v_xy[1:-1,2:], dx[:-1,1:-1], params)

    #--------------------------------------------------------------------------
    #--------------------------- jv in the y-direction ------------------------
    #--------------------------------------------------------------------------
    # These are obtained by changing 1 by N and N by 1 in the definitions for the
    # x-direction: v_sm1 -> v_smN, v_spN -> v_sp1.
    # jvn_s in the y-direction only for the inner part of the system
    jvny_s = mu[1:-1,1:-1] * get_jvn_s(efn_xy[1:-1,1:-1], efn_xy[1:-1,2:], \
    v_xy[1:-1,:-2], v_xy[1:-1,1:-1], v_xy[1:-1,2:], v_xy[:-2,1:-1],\
    v_xy[2:,1:-1], dy[1:-1,1:], params)

    # jvn_smN in the y-direction only for the inner part of the system
    jvny_smN = mu[1:-1,:-2] * get_jvn_sm1(efn_xy[1:-1,:-2], efn_xy[1:-1,1:-1], \
    v_xy[1:-1,:-2], v_xy[1:-1,1:-1], v_xy[1:-1,2:], v_xy[:-2,1:-1],\
    v_xy[2:,1:-1], dy[1:-1,:-1], params)

    # jvp_s in the y-direction only for the inner part of the system
    jvpy_s = mu[1:-1,1:-1] * get_jvp_s(efp_xy[1:-1,1:-1], efp_xy[1:-1,2:], \
    v_xy[1:-1,:-2], v_xy[1:-1,1:-1], v_xy[1:-1,2:], v_xy[:-2,1:-1],\
    v_xy[2:,1:-1], dy[1:-1,1:], params)

    # jvp_smN in the y-direction only for the inner part of the system
    jvpy_smN = mu[1:-1,:-2] * get_jvp_sm1(efp_xy[1:-1,:-2], efp_xy[1:-1,1:-1], \
    v_xy[1:-1,:-2], v_xy[1:-1,1:-1], v_xy[1:-1,2:], v_xy[:-2,1:-1],\
    v_xy[2:,1:-1], dy[1:-1,:-1], params)

    #--------------------------------------------------------------------------
    #---------------------------------- uv ------------------------------------
    #--------------------------------------------------------------------------
    # I defined U = G - R
    # uvn
    uvn = get_uvn(n_xy[1:-1,1:-1], p_xy[1:-1,1:-1], v_xy[1:-1,:-2],\
    v_xy[:-2,1:-1], v_xy[1:-1,1:-1], v_xy[2:,1:-1], v_xy[1:-1,2:], \
    g_xy[1:-1,1:-1], S_xy[1:-1,1:-1], SGB_xy[1:-1,1:-1], params)

    # uvp 
    uvp = get_uvp(n_xy[1:-1,1:-1], p_xy[1:-1,1:-1], v_xy[1:-1,:-2],
    v_xy[:-2,1:-1], v_xy[1:-1,1:-1], v_xy[2:,1:-1], v_xy[1:-1,2:], \
    g_xy[1:-1,1:-1], S_xy[1:-1,1:-1], SGB_xy[1:-1,1:-1], params)

    #--------------------------------------------------------------------------
    #------------------------------ fn ----------------------------------------
    #--------------------------------------------------------------------------
    dxbar = (dx[1:,1:-1] + dx[:-1,1:-1]) / 2.
    dybar = (dy[1:-1,1:] + dy[1:-1,:-1]) / 2.

    fn = (jvnx_s - jvnx_sm1) / dxbar + (jvny_s - jvny_smN) / dybar + uvn

    # reshape the arrays as 1D arrays
    fn = (fn.T).reshape((Nx-2)*(Ny-2))

    # update the vector rows for the inner part of the system
    fn_rows = [3*s for s in sites]
    vec[fn_rows] = fn

    #--------------------------------------------------------------------------
    #------------------------------ fp ----------------------------------------
    #--------------------------------------------------------------------------
    fp = (jvpx_s - jvpx_sm1) / dxbar + (jvpy_s - jvpy_smN) / dybar + uvp

    # reshape the arrays as 1D arrays
    fp = (fp.T).reshape((Nx-2)*(Ny-2))

    # update the vector rows for the inner part of the system
    fp_rows = [3*s+1 for s in sites]
    vec[fp_rows] = fp

    #--------------------------------------------------------------------------
    #------------------------------ fv ----------------------------------------
    #--------------------------------------------------------------------------
    # local charge density
    fGB = (n_xy[1:-1,1:-1] + pGB) / (n_xy[1:-1,1:-1] + p_xy[1:-1,1:-1] + nGB + pGB)
    rhoGB = NGB_xy[1:-1,1:-1] / 2. * (1 - 2*fGB)

    fv = ((v_xy[1:-1, 1:-1] - v_xy[:-2, 1:-1]) / dx[:-1,1:-1]\
         -(v_xy[2:, 1:-1] - v_xy[1:-1, 1:-1]) / dx[1:,1:-1]) / dxbar\
         +((v_xy[1:-1, 1:-1] - v_xy[1:-1, :-2]) / dy[1:-1,:-1]\
         -(v_xy[1:-1, 2:] - v_xy[1:-1, 1:-1]) / dy[1:-1,1:]) / dybar\
         -(rho_xy[1:-1, 1:-1] + rhoGB + p_xy[1:-1, 1:-1] - n_xy[1:-1, 1:-1])

    # reshape the arrays as 1D arrays
    fv = (fv.T).reshape((Nx-2)*(Ny-2))

    # update the vector rows for the inner part of the system
    fv_rows = [3*s+2 for s in sites]
    vec[fv_rows] = fv


    ###########################################################################
    #                  left boundary: i = 0 and 0 < j < Ny-1                  #
    ###########################################################################
    # list of the sites on the left side
    sites = [j*Nx for j in range(1,Ny-1)]

    # compute an, ap, av
    an = mu[0,1:-1] * (v_xy[0,1:-1] - v_xy[1,1:-1]) / dx[0,1:-1] * \
                      (exp(efn_xy[1,1:-1]) - exp(efn_xy[0,1:-1]))\
         - (exp(-v_xy[1,1:-1]) - exp(-v_xy[0,1:-1])) * scn[0] * (n_xy[0,1:-1] - nD)
    ap = mu[0,1:-1] * (v_xy[0,1:-1] - v_xy[1,1:-1]) / dx[0,1:-1] * \
                      (exp(efp_xy[1,1:-1]) - exp(efp_xy[0,1:-1]))\
         - (exp(v_xy[1,1:-1]) - exp(v_xy[0,1:-1])) * scp[0] * (p_xy[0,1:-1] - ni**2/nD)
    av = 0 # to ensure Dirichlet BCs

    # update the vector rows
    an_rows = [3*s for s in sites]
    ap_rows = [3*s+1 for s in sites]
    av_rows = [3*s+2 for s in sites]
    vec[an_rows] = an
    vec[ap_rows] = ap
    vec[av_rows] = av

    ###########################################################################
    #                right boundary: i = Nx-1 and 0 < j < Ny-1                #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + j*Nx for j in range(1,Ny-1)]

    # recombination rates
    r = S_xy[Nx-1,1:-1] * (n_xy[Nx-1,1:-1]*p_xy[Nx-1,1:-1]-ni**2) / \
                          (n_xy[Nx-1,1:-1]+p_xy[Nx-1,1:-1]+n1+p1)
    rGB = SGB_xy[Nx-1,1:-1] * (n_xy[Nx-1,1:-1]*p_xy[Nx-1,1:-1]-ni**2) / \
                          (n_xy[Nx-1,1:-1]+p_xy[Nx-1,1:-1]+nGB+pGB)

    # compute bn, bp, bv
    dxbar = dx[-1,1:-1]
    dybar = (dy[-1,1:-1] + dy[-1,:-2]) / 2.

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

    bn = mu[-2,1:-1] * (v_sm1 - v_s) /dx[-1,1:-1] *  (exp(efn_s) - exp(efn_sm1))\
         + dxbar * ((r + rGB - g_xy[-1,1:-1]) * (exp(-v_spN) - exp(-v_s)) * \
         (exp(-v_s) - exp(-v_sm1)) * (exp(-v_s) - exp(-v_smN)) - \
         mu[-1,1:-1]/(dybar*dy[-1,1:]) * (v_s - v_spN) * (exp(efn_spN) -\
         exp(efn_s)) * (exp(-v_s) - exp(-v_sm1)) * (exp(-v_s) - exp(-v_smN))\
         + mu[-1,:-2]/(dybar*dy[-1,:-1]) * (v_smN - v_s) * (exp(efn_s) -\
         exp(efn_smN)) * (exp(-v_s) - exp(-v_sm1)) * (exp(-v_spN) - exp(-v_s)))\
         + scn[1] * (n_xy[-1,1:-1] - ni**2 / nA) * (exp(-v_s) - exp(-v_sm1))\
         * (exp(-v_spN) - exp(-v_s)) * (exp(-v_s) - exp(-v_smN))

    bp = mu[-2,1:-1] * (v_sm1 - v_s) /dx[-1,1:-1] *  (exp(efp_s) - exp(efp_sm1))\
         + dxbar * ((g_xy[-1,1:-1] - r - rGB) * (exp(v_spN) - exp(v_s)) * \
         (exp(v_s) - exp(v_sm1)) * (exp(v_s) - exp(v_smN)) - \
         mu[-1,1:-1]/(dybar*dy[-1,1:]) * (v_s - v_spN) * (exp(efp_spN) -\
         exp(efp_s)) * (exp(v_s) - exp(v_sm1)) * (exp(v_s) - exp(v_smN))\
         + mu[-1,:-2]/(dybar*dy[-1,:-1]) * (v_smN - v_s) * (exp(efp_s) -\
         exp(efp_smN)) * (exp(v_s) - exp(v_sm1)) * (exp(v_spN) - exp(v_s)))\
         - scp[1] * (p_xy[-1,1:-1] - nA) * (exp(v_s) - exp(v_sm1)) \
         * (exp(v_spN) - exp(v_s)) * (exp(v_s) - exp(v_smN))

    bv = 0

    # update the vector rows
    bn_rows = [3*s for s in sites]
    bp_rows = [3*s+1 for s in sites]
    bv_rows = [3*s+2 for s in sites]
    vec[bn_rows] = bn
    vec[bp_rows] = bp
    vec[bv_rows] = bv      

    ###########################################################################
    #                         top and bottom boundaries                       #
    ###########################################################################
    # I want Jy=0 => defn = defp = 0. In addition, a zero is needed in vec for
    # v_s to ensure that the last 2 rows are equal.

    # The same applied for the bottom layer. Since vec is created with zeros in
    # it, there is nothing to do.
   
    return vec
