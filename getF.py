import numpy as np
from numpy import exp
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

from sesame.observables import *

def getFandJ(v, efn, efp, params):

    bl, eg, nC, nV, nA, nD, scn, scp, g, mu, tau, rho,\
    NGB, tauGB, nGB, pGB,\
    n1, p1, ni, xpts, ypts = params

    dx = xpts[1:] - xpts[:-1]
    dy = ypts[1:] - ypts[:-1]
    
    Nx = xpts.shape[0]
    Ny = ypts.shape[0]
    vec = np.zeros((3*Nx*Ny,), dtype=float)

    rows = []
    columns = []
    data = []

    # reshape the vectors to conform to [x,y] coordinates
    v_xy = v.reshape(Ny, Nx).T
    efn_xy = efn.reshape(Ny, Nx).T
    efp_xy = efp.reshape(Ny, Nx).T
    mu = mu.reshape(Ny, Nx).T

    # currents in the x-direction at all sites but the last column
    jn_x = mu * get_jn(efn_xy[:-1,:], efn_xy[1:,:], v_xy[:-1,:], v[1:,:], dx, params)
    jp_x = mu * get_jp(efp_xy[:-1,:], efp_xy[1:,:], v_xy[:-1,:], v[1:,:], dx, params)
    # currents in the y-direction at all sites but the last row
    jn_y = mu * get_jn(efn_xy[:,:-1], efn_xy[:,1:], v_xy[:,:-1], v[:,1:], dx, params)
    jp_y = mu * get_jp(efp_xy[:,:-1], efp_xy[:,1:], v_xy[:,:-1], v[:,1:], dx, params)
    # current derivatives
    djn_x = mu * get_jn_derivs(efn_xy[:-1,:], efn_xy[1:,:], v_xy[:-1,:], v[1:,:], dx, params)
    djp_x = mu * get_jp_derivs(efp_xy[:-1,:], efp_xy[1:,:], v_xy[:-1,:], v[1:,:], dx, params)
    djn_y = mu * get_jn_derivs(efn_xy[:,:-1], efn_xy[:,1:], v_xy[:,:-1], v[:,1:], dx, params)
    djp_y = mu * get_jp_derivs(efp_xy[:,:-1], efp_xy[:,1:], v_xy[:,:-1], v[:,1:], dx, params)

    # for c in range(1, xpts.shape[0]-1, 1):
    for s in range(Nx*Ny):
        j = s//Nx
        i = s - j*Nx

        # rows
        fn_row = 3*s
        fp_row = 3*s+1
        fv_row = 3*s+2
        
        # columns
        efn_smN_col = 3*(s-Nx)
        efn_sm1_col = 3*(s-1)
        efn_s_col = 3*s
        efn_sp1_col = 3*(s+1)
        efn_spN_col = 3*(s+Nx)
        
        efp_smN_col = 3*(s-Nx)+1
        efp_sm1_col = 3*(s-1)+1
        efp_s_col = 3*s+1
        efp_sp1_col = 3*(s+1)+1
        efp_spN_col = 3*(s+Nx)+1
        
        v_smN_col = 3*(s-Nx)+2
        v_sm1_col = 3*(s-1)+2
        v_s_col = 3*s+2
        v_sp1_col = 3*(s+1)+2
        v_spN_col = 3*(s+Nx)+2

        # values of the guess for efn, efp, v
        efn_s = efn[s]
        efp_s = efp[s]
        v_s = v[s]
        mu_s = mu[s]

        if s-Nx > 0:
            efn_smN = efn[s-Nx]
            efp_smN = efp[s-Nx]
            v_smN = v[s-Nx]
            mu_smN = mu[s-Nx]

        if s > 0:
            efn_sm1 = efn[s-1]
            efp_sm1 = efp[s-1]
            v_sm1 = v[s-1]
            mu_sm1 = mu[s-1]

        if s+1 < Nx*Ny:
            efn_sp1 = efn[s+1]
            efp_sp1 = efp[s+1]
            v_sp1 = v[s+1]

        if s+Nx < Nx*Ny:
            efn_spN = efn[s+Nx]
            efp_spN = efp[s+Nx]
            v_spN = v[s+Nx]


        n_s = get_n(efn_s, v_s, params)
        p_s = get_p(efp_s, v_s, params)
        
        if s in NGB:
            # GB charge density
            fGB = (n_s + pGB) / (n_s + p_s + nGB + pGB)
            rhoGB = NGB[s]/2. * (1 - 2*fGB)
            drhoGB_dv = -NGB[s] * (n_s*(n_s+p_s+nGB+pGB)-(n_s+pGB)*(n_s-p_s))\
                                / (n_s+p_s+nGB+pGB)**2
            drhoGB_defn = -NGB[s] * (n_s*(n_s+p_s+nGB+pGB)-(n_s+pGB)*n_s)\
                                  / (n_s+p_s+nGB+pGB)**2
            drhoGB_defp = NGB[s] * (n_s+pGB)*p_s / (n_s+p_s+nGB+pGB)**2
            # GB recombination rate
            rGB = get_rr(efn_s, efp_s, v_s, nGB, pGB, tauGB[s], params)
            drrGB_defp, drrGB_defn, drrGB_dv = \
            get_rr_derivs(efn_s, efp_s, v_s, nGB, pGB, tauGB[s], params)
        else:
            rhoGB = 0
            drhoGB_dv, drhoGB_defn, drhoGB_defp = 0, 0, 0
            rGB = 0
            drrGB_defp, drrGB_defn, drrGB_dv = 0, 0, 0


        ## recombination rate and its derivatives (needed everywhere)
        #################################################################
        r = get_rr(efn_s, efp_s, v_s, n1, p1, tau[s], params) + rGB
        
        drr_defp_s, drr_defn_s, drr_dv_s = \
        get_rr_derivs(efn_s, efp_s, v_s, n1, p1, tau[s], params)\

        drr_defp_s += drrGB_defp
        drr_defn_s += drrGB_defn
        drr_dv_s += drrGB_dv


        ## inside the grid
        if 0 < i < Nx-1 and 0 < j < Ny-1:
            # spacing
            dx_i = dx[i]
            dy_j = dy[j]
            dx_im1 = dx[i-1]
            dxbar = (dx_i + dx_im1)/2.
            dy_jm1 = dy[j-1]
            dybar = (dy_j + dy_jm1)/2.

            ## f values for equations governing efn, efp, v.
            ########################################################################
            fn = (g[s] - r)\
                 + mu_s / dxbar * get_jn(efn_s, efn_sp1, v_s, v_sp1, dx_i, params)\
                 - mu_sm1 / dxbar * get_jn(efn_sm1, efn_s, v_sm1, v_s, dx_im1, params)\
                 + mu_s / dybar * get_jn(efn_s, efn_spN, v_s, v_spN, dy_j, params)\
                 - mu_smN / dybar * get_jn(efn_smN, efn_s, v_smN, v_s, dy_jm1, params)

            fp = (r - g[s])\
                 + mu_s / dxbar * get_jp(efp_s, efp_sp1, v_s, v_sp1, dx_i, params)\
                 - mu_sm1 / dxbar * get_jp(efp_sm1, efp_s, v_sm1, v_s, dx_im1, params)\
                 + mu_s / dybar * get_jp(efp_s, efp_spN, v_s, v_spN, dy_j, params)\
                 - mu_smN / dybar * get_jp(efp_smN, efp_s, v_smN, v_s, dy_jm1, params)

            fv = 1./dxbar * ((v_s-v_sm1)/dx_im1 - (v_sp1-v_s)/dx_i)\
                 + 1./dybar * ((v_s-v_smN)/dy_jm1 - (v_spN-v_s)/dy_j)\
                 - (rho[s] + rhoGB + nV*exp(bl-eg+efp_s-v_s) - nC*exp(-bl+efn_s+v_s))

                        
            ## right-hand side vector
            ##################################
            vec[fn_row] = fn
            vec[fp_row] = fp
            vec[fv_row] = fv
        # inside the grid completed

        # left boundary
        elif i == 0 and 0 < j < Ny-1:
            # spacing
            dx_i = dx[i]

            # currents and densities on the left side
            jnx = mu_s * get_jn(efn_s, efn_sp1, v_s, v_sp1, dx_i, params)
            jpx = mu_s * get_jp(efp_s, efp_sp1, v_s, v_sp1, dx_i, params)
 
            # a_n, a_p values, a_v
            vec[fn_row] = jnx - scn[0] * (n_s - nD)
            vec[fp_row] = jpx + scp[0] * (p_s - ni**2 / nD)
            vec[fv_row] = 0 # Dirichlet BC for v

            
        # right boundary
        elif i == Nx-1 and 0 < j < Ny-1:
            # spacing
            dx_im1 = dx[-1]
            dxbar = dx_im1
            dy_jm1 = dy[-1]
            dy_j = dy[j]
            dybar = (dy_j + dy_jm1)/2.

            # currents and densities on the right side
            jnx_sm1 = mu_sm1 * get_jn(efn_sm1, efn_s, v_sm1, v_s, dx_im1, params)
            jny_s = mu_s * get_jn(efn_s, efn_spN, v_s, v_spN, dy_j, params)
            jny_smN = mu_smN * get_jn(efn_smN, efn_s, v_smN, v_s, dy_jm1, params)
            jnx_s = jnx_sm1 + dxbar * (r - g[s] - (jny_s - jny_smN)/dybar)

            jpx_sm1 = mu_sm1 * get_jp(efp_sm1, efp_s, v_sm1, v_s, dx_im1, params)
            jpy_s = mu_s * get_jp(efp_s, efp_spN, v_s, v_spN, dy_j, params)
            jpy_smN = mu_smN * get_jp(efp_smN, efp_s, v_smN, v_s, dy_j, params)
            jpx_s = jpx_sm1 + dxbar * (g[s] - r - (jpy_s - jpy_smN)/dybar)

 
            # b_n, b_p and b_v values
            vec[fn_row] = jnx_s + scn[1] * (n_s - ni**2 / nA)
            vec[fp_row] = jpx_s - scp[1] * (p_s - nA)
            vec[fv_row] = 0

           
        # top boundary
        if j == Ny-1:
            # I want Jy=0 => defn = defp = 0
            vec[fn_row] = 0
            vec[fp_row] = 0
            vec[fv_row] = 0

            # top_n
            ######################################
            rows += [fn_row, fn_row]
            columns += [efn_smN_col, efn_s_col]
            data += [1, -1]

            # top_p
            ######################################
            rows += [fp_row, fp_row]
            columns += [efp_smN_col, efp_s_col]
            data += [1, -1]

            ## top_v
            ######################################
            rows += [fv_row, fv_row]
            columns += [v_smN_col, v_s_col]
            data += [1, -1]

        # bottom boundary
        if j == 0:
            # I want Jy=0 => defn = defp = 0
            vec[fn_row] = 0
            vec[fp_row] = 0
            vec[fv_row] = 0

    return vec
