import numpy as np

from sesame.observables import *

def getF(v, efn, efp, params):
    bl, eg, nC, nV, nA, nD, scn, scp, g, mu, tau, rho,\
    NGB, SGB, nGB, pGB,\
    n1, p1, ni, xpts, ypts, eps = params

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

    # right hand side vector
    vec = np.zeros((3*Nx*Ny,))

    ###########################################################################
    #               organization of the right hand side vector                #
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
    

    ###########################################################################
    #       inside the system: 0 < i < Nx-1 and 0 < j < Ny-1                  #
    ###########################################################################
    # We compute fn, fp, fv. Those functions are only defined on the
    # inner part of the system. All the edges containing boundary conditions.

    # list of the sites inside the system
    sites = [i + j*Nx for j in range(1,Ny-1) for i in range(1,Nx-1)]

    # dxbar and dybar
    dxbar = (dx[1:,1:-1] + dx[:-1,1:-1]) / 2.
    dybar = (dy[1:-1,1:] + dy[1:-1,:-1]) / 2.

    # gether efn, efp, v for all relevant sites
    efn_smN = efn_xy[1:-1,:-2]
    efn_sm1 = efn_xy[:-2,1:-1]
    efn_s = efn_xy[1:-1,1:-1]
    efn_sp1 = efn_xy[2:,1:-1]
    efn_spN = efn_xy[1:-1,2:]

    efp_smN = efp_xy[1:-1,:-2]
    efp_sm1 = efp_xy[:-2,1:-1]
    efp_s = efp_xy[1:-1,1:-1]
    efp_sp1 = efp_xy[2:,1:-1]
    efp_spN = efp_xy[1:-1,2:]

    v_smN = v_xy[1:-1,:-2]
    v_sm1 = v_xy[:-2,1:-1]
    v_s = v_xy[1:-1,1:-1]
    v_sp1 = v_xy[2:,1:-1]
    v_spN = v_xy[1:-1,2:]

    # compute the currents
    jnx_s = mu[1:-1,1:-1] * get_jn(efn_s, efn_sp1, v_s, v_sp1, dx[1:,1:-1], params)
    jnx_sm1 = mu[:-2,1:-1] * get_jn(efn_sm1, efn_s, v_sm1, v_s, dx[:-1,1:-1], params)
    jny_s = mu[1:-1,1:-1] * get_jn(efn_s, efn_spN, v_s, v_spN, dy[1:-1,1:], params)
    jny_smN = mu[1:-1,:-2] * get_jn(efn_smN, efn_s, v_smN, v_s, dy[1:-1,:-1], params)

    jpx_s = mu[1:-1,1:-1] * get_jp(efp_s, efp_sp1, v_s, v_sp1, dx[1:,1:-1], params)
    jpx_sm1 = mu[:-2,1:-1] * get_jp(efp_sm1, efp_s, v_sm1, v_s, dx[:-1,1:-1], params)
    jpy_s = mu[1:-1,1:-1] * get_jp(efp_s, efp_spN, v_s, v_spN, dy[1:-1,1:], params)
    jpy_smN = mu[1:-1,:-2] * get_jp(efp_smN, efp_s, v_smN, v_s, dy[1:-1,:-1], params)

    # recombination rates
    r = get_rr(n_xy[1:-1,1:-1], p_xy[1:-1,1:-1], n1, p1, S_xy[1:-1,1:-1], params)
    rGB = get_rr(n_xy[1:-1,1:-1], p_xy[1:-1,1:-1], nGB, pGB, SGB_xy[1:-1,1:-1], params)

    
    #--------------------------------------------------------------------------
    #------------------------------ fn ----------------------------------------
    #--------------------------------------------------------------------------
    fn = (jnx_s - jnx_sm1) / dxbar + (jny_s - jny_smN) / dybar +\
          g_xy[1:-1,1:-1] - r - rGB

    # reshape the arrays as 1D arrays
    fn = (fn.T).reshape((Nx-2)*(Ny-2))

    # update the vector rows for the inner part of the system
    fn_rows = [3*s for s in sites]
    vec[fn_rows] = fn

    #--------------------------------------------------------------------------
    #------------------------------ fp ----------------------------------------
    #--------------------------------------------------------------------------
    fp = (jpx_s - jpx_sm1) / dxbar + (jpy_s - jpy_smN) / dybar +\
           r + rGB - g_xy[1:-1,1:-1]

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
         -(rho_xy[1:-1, 1:-1] + rhoGB + p_xy[1:-1, 1:-1] - n_xy[1:-1, 1:-1])/eps


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

    # compute the currents
    jnx = mu[0,1:-1] * get_jn(efn_xy[0,1:-1], efn_xy[1,1:-1], v_xy[0,1:-1],
                              v_xy[1,1:-1], dx[0,1:-1], params)
    jpx = mu[0,1:-1] * get_jp(efp_xy[0,1:-1], efp_xy[1,1:-1], v_xy[0,1:-1],
                              v_xy[1,1:-1], dx[0,1:-1], params)

    # compute an, ap, av
    n_eq = 0
    p_eq = 0
    if rho_xy[0, 1] < 0: # p doped
        p_eq = -rho_xy[0, 1]
        n_eq = ni**2 / p_eq
    else: # n doped
        n_eq = rho_xy[0, 1]
        p_eq = ni**2 / n_eq
        
    an = jnx - scn[0] * (n_xy[0,1:-1] - n_eq)
    ap = jpx + scp[0] * (p_xy[0,1:-1] - p_eq)
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

    # dxbar and dybar
    dxbar = dx[-1,1:-1]
    dybar = (dy[-1,1:] + dy[-1,:-1]) / 2.

    # gather efn, efp, v
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

    # recombination rates
    r = get_rr(n_xy[Nx-1,1:-1], p_xy[Nx-1,1:-1], n1, p1, S_xy[Nx-1,1:-1], params)
    rGB = get_rr(n_xy[Nx-1,1:-1], p_xy[Nx-1,1:-1], nGB, pGB, SGB_xy[Nx-1,1:-1], params)

    # compute the currents
    jnx_sm1 = mu[-2,1:-1] * get_jn(efn_sm1, efn_s, v_sm1, v_s, dx[-1,1:-1], params)
    jny_s = mu[-1,1:-1] * get_jn(efn_s, efn_spN, v_s, v_spN, dy[-1,1:], params)
    jny_smN = mu[-1,:-2] * get_jn(efn_smN, efn_s, v_smN, v_s, dy[-1,:-1], params)
    jnx_s = jnx_sm1 + dxbar * (r + rGB - g_xy[-1,1:-1] - (jny_s - jny_smN)/dybar)

    jpx_sm1 = mu[-2,1:-1] * get_jp(efp_sm1, efp_s, v_sm1, v_s, dx[-1,1:-1], params)
    jpy_s = mu[-1,1:-1] * get_jp(efp_s, efp_spN, v_s, v_spN, dy[-1,1:], params)
    jpy_smN = mu[-1,:-2] * get_jp(efp_smN, efp_s, v_smN, v_s, dy[-1,:-1], params)
    jpx_s = jpx_sm1 + dxbar * (g_xy[-1,1:-1] - r - rGB - (jpy_s - jpy_smN)/dybar)

    # b_n, b_p and b_v values
    n_eq = 0
    p_eq = 0
    if rho_xy[-1, 1] < 0: # p doped
        p_eq = -rho_xy[-1, 1]
        n_eq = ni**2 / p_eq
    else: # n doped
        n_eq = rho_xy[-1, 1]
        p_eq = ni**2 / n_eq
        
    bn = jnx_s + scn[1] * (n_xy[-1,1:-1] - n_eq)
    bp = jpx_s - scp[1] * (p_xy[-1,1:-1] - p_eq)
    bv = 0 # Dirichlet BC

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
