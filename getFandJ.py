import numpy as np
from numpy import exp
from scipy.sparse import csr_matrix

from sesame.observables import *

def getFandJ(v, efn, efp, params):

    bl, eg, nC, nV, nA, nD, scn, scp, g, mu, tau, rho, xpts, ypts = params
    dx = xpts[1:] - xpts[:-1]
    dy = ypts[1:] - ypts[:-1]
    
    Nx = xpts.shape[0]
    Ny = ypts.shape[0]
    vec = np.zeros((3*Nx*Ny,), dtype=float)

    rows = []
    columns = []
    data = []

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


        ## recombination rate and its derivatives (needed everywhere)
        #################################################################
        r = get_rr(efn_s, efp_s, v_s, tau[s], params)
        
        drr_defp_s, drr_defn_s, drr_dv_s = \
        get_rr_derivs(efn_s, efp_s, v_s, tau[s], params)


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
                 - (rho[s] + nV*exp(bl-eg+efp_s-v_s) - nC*exp(-bl+efn_s+v_s))

            ## fn derivatives
            ###################################################
            # get the derivatives of jx_s, jx_sm1, jy_s, jy_smN
            djx_s_defn_s, djx_s_defn_sp1, djx_s_dv_s, djx_s_dv_sp1 = \
            get_jn_derivs(efn_s, efn_sp1, v_s, v_sp1, dx_i, params)

            djy_s_defn_s, djy_s_defn_spN, djy_s_dv_s, djy_s_dv_spN = \
            get_jn_derivs(efn_s, efn_spN, v_s, v_spN, dy_j, params)

            djx_sm1_defn_sm1, djx_sm1_defn_s, djx_sm1_dv_sm1, djx_sm1_dv_s = \
            get_jn_derivs(efn_sm1, efn_s, v_sm1, v_s, dx_im1, params)

            djy_smN_defn_smN, djy_smN_defn_s, djy_smN_dv_smN, djy_smN_dv_s = \
            get_jn_derivs(efn_smN, efn_s, v_smN, v_s, dy_jm1, params)

            # compute the derivatives of fn
            defn_smN = -mu_smN / dybar * djy_smN_defn_smN
            dv_smN = -mu_smN / dybar * djy_smN_dv_smN
            defn_sm1 = -mu_sm1 / dxbar * djx_sm1_defn_sm1
            dv_sm1 = -mu_sm1 / dxbar * djx_sm1_dv_sm1
            defn_s = mu_s * (djx_s_defn_s / dxbar + djy_s_defn_s / dybar)\
                     - mu_sm1 / dxbar * djx_sm1_defn_s - mu_smN / dybar * djy_smN_defn_s\
                     - drr_defn_s
            defp_s = -drr_defp_s
            dv_s = mu_s * (djx_s_dv_s / dxbar + djy_s_dv_s / dybar)\
                     - mu_sm1 / dxbar * djx_sm1_dv_s - mu_smN / dybar * djy_smN_dv_s\
                     - drr_dv_s
            defn_sp1 = mu_s / dxbar * djx_s_defn_sp1
            dv_sp1 = mu_s / dxbar * djx_s_dv_sp1
            defn_spN = mu_s / dybar * djy_s_defn_spN
            dv_spN = mu_s / dybar * djy_s_dv_spN

            # keep track of row and column indices, and store the values
            rows += 11*[fn_row]
            columns += [efn_smN_col, v_smN_col, efn_sm1_col, v_sm1_col,\
                        efn_s_col, efp_s_col, v_s_col,\
                        efn_sp1_col, v_sp1_col, efn_spN_col, v_spN_col]
            data += [defn_smN, dv_smN, defn_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                     defn_sp1, dv_sp1, defn_spN, dv_spN]


            ## fp derivatives
            ###################################################
            # get the derivatives of jx_s, jx_sm1, jy_s, jy_smN
            djx_s_defp_s, djx_s_defp_sp1, djx_s_dv_s, djx_s_dv_sp1 = \
            get_jp_derivs(efp_s, efp_sp1, v_s, v_sp1, dx_i, params)

            djy_s_defp_s, djy_s_defp_spN, djy_s_dv_s, djy_s_dv_spN = \
            get_jp_derivs(efp_s, efp_spN, v_s, v_spN, dy_j, params)

            djx_sm1_defp_sm1, djx_sm1_defp_s, djx_sm1_dv_sm1, djx_sm1_dv_s = \
            get_jp_derivs(efp_sm1, efp_s, v_sm1, v_s, dx_im1, params)

            djy_smN_defp_smN, djy_smN_defp_s, djy_smN_dv_smN, djy_smN_dv_s =  \
            get_jp_derivs(efp_smN, efp_s, v_smN, v_s, dy_jm1, params)

            # compute the derivatives of fp
            defp_smN = -mu_smN / dybar * djy_smN_defp_smN
            dv_smN = -mu_smN / dybar * djy_smN_dv_smN
            defp_sm1 = -mu_sm1 / dxbar * djx_sm1_defp_sm1
            dv_sm1 = -mu_sm1 / dxbar * djx_sm1_dv_sm1
            defn_s = drr_defn_s
            defp_s = mu_s * (djx_s_defp_s / dxbar + djy_s_defp_s / dybar)\
                     - mu_sm1 / dxbar * djx_sm1_defp_s - mu_smN / dybar * djy_smN_defp_s\
                     + drr_defp_s
            dv_s = mu_s * (djx_s_dv_s / dxbar + djy_s_dv_s / dybar)\
                     - mu_sm1 / dxbar * djx_sm1_dv_s - mu_smN / dybar * djy_smN_dv_s\
                     + drr_dv_s
            defp_sp1 = mu_s / dxbar * djx_s_defp_sp1
            dv_sp1 = mu_s / dxbar * djx_s_dv_sp1
            defp_spN = mu_s / dybar * djy_s_defp_spN
            dv_spN = mu_s / dybar * djy_s_dv_spN

            # keep track of row and column indices, and store the values
            rows += 11*[fp_row]
            columns += [efp_smN_col, v_smN_col, efp_sm1_col, v_sm1_col,\
                        efn_s_col, efp_s_col, v_s_col,\
                        efp_sp1_col, v_sp1_col, efp_spN_col, v_spN_col]
            data += [defp_smN, dv_smN, defp_sm1, dv_sm1, defn_s, defp_s, dv_s,\
                     defp_sp1, dv_sp1, defp_spN, dv_spN]


            ## fv derivatives
            ###################################################
            dfv_dvmN = -1./(dy_jm1 * dybar )
            dfv_dvm1 = -1./(dx_im1 * dxbar)
            dfv_dv = 2./(dx_i * dx_im1) + 2./(dy_j * dy_jm1) +\
                     nV*exp(bl-eg+efp_s-v_s) + nC*exp(-bl+efn_s+v_s)
            dfv_defn = nC*exp(-bl+efn_s+v_s)
            dfv_defp = -nV*exp(bl-eg+efp_s-v_s)
            dfv_dvp1 = -1./(dx_i * dxbar)
            dfv_dvpN = -1./(dy_j * dybar)

            rows += 7*[fv_row]
            columns += [v_smN_col, v_sm1_col, efn_s_col, efp_s_col, v_s_col,\
                        v_sp1_col, v_spN_col]
            data += [dfv_dvmN, dfv_dvm1, dfv_defn, dfv_defp, dfv_dv, dfv_dvp1, dfv_dvpN]
            
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
            n_s = get_n(efn_s, v_s, params)
            p_s = get_p(efp_s, v_s, params)
 
            # a_n, a_p values, a_v
            vec[fn_row] = jnx - scn[0] * (n_s - nD)
            vec[fp_row] = jpx + scp[0] * (p_s - exp(-eg)/nD)
            vec[fv_row] = 0 # Dirichlet BC for v

            ## a_n derivatives on the left boundary
            ############################################
            defn_s, defn_sp1, dv_s, dv_sp1 = get_jn_derivs(efn_s, efn_sp1, v_s,
                                                           v_sp1, dx_i, params)
            rows += 4*[fn_row]
            columns += [efn_s_col, v_s_col, efn_sp1_col, v_sp1_col]
            data += [mu_s * defn_s - scn[0] * n_s, mu_s * dv_s - scn[0] * n_s,\
                     mu_s * defn_sp1, mu_s * dv_sp1]
        
            ## a_p derivatives on the left boundary
            ##############################################
            defp_s, defp_sp1, dv_s, dv_sp1 = get_jp_derivs(efp_s, efp_sp1, v_s,
                                                           v_sp1, dx_i, params)
            rows += 4*[fp_row]
            columns += [efp_s_col, v_s_col, efp_sp1_col, v_sp1_col]
            data += [mu_s * defp_s + scp[0] * p_s, mu_s * dv_s - scp[0] * p_s,\
                     mu_s * defp_sp1, mu_s * dv_sp1]

            ## a_v derivative
            #######################
            rows.append(fv_row)
            columns.append(v_s_col)
            data.append(1)

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

            n_s = get_n(efn_s, v_s, params)
            p_s = get_p(efp_s, v_s, params)
 
            # b_n, b_p and b_v values
            vec[fn_row] = jnx_s + scn[1] * (n_s - exp(-eg) / nA)
            vec[fp_row] = jpx_s - scp[1] * (p_s - nA)
            vec[fv_row] = 0

            ## b_n derivatives on the right boundary
            ############################################
            djnx_sm1_defn_sm1, djnx_sm1_defn_s, djnx_sm1_dv_sm1, djnx_sm1_dv_s =\
            get_jn_derivs(efn_sm1, efn_s, v_sm1, v_s, dx_im1, params)

            djny_s_defn_s, djny_s_defn_spN, djny_s_dv_s, djny_s_dv_spN = \
            get_jn_derivs(efn_s, efn_spN, v_s, v_spN, dy_j, params)

            djny_smN_defn_smN, djny_smN_defn_s, djny_smN_dv_smN, djny_smN_dv_s = \
            get_jn_derivs(efn_smN, efn_s, v_smN, v_s, dy_jm1, params)

            rows += 9*[fn_row]
            columns += [efn_smN_col, v_smN_col, efn_sm1_col, v_sm1_col, efn_s_col,\
                        efp_s_col, v_s_col, efn_spN_col, v_spN_col]

            data += [mu_smN * dxbar/dybar * djny_smN_defn_smN, \
                     mu_smN * dxbar/dybar * djny_smN_dv_smN, \
                     mu_sm1 * djnx_sm1_defn_sm1, mu_sm1 * djnx_sm1_dv_sm1, \
                     mu_sm1 * djnx_sm1_defn_s + dxbar * (drr_defn_s - (mu_s * djny_s_defn_s \
                     - mu_smN * djny_smN_defn_s) / dybar) + scn[1] * n_s, \
                     dxbar * drr_defp_s,\
                     mu_sm1 * djnx_sm1_dv_s + dxbar * (drr_dv_s - (mu_s * djny_s_dv_s \
                     - mu_smN * djny_smN_dv_s) / dybar) + scn[1] * n_s, \
                     -mu_s * dxbar/dybar * djny_s_defn_spN, -mu_s * dxbar/dybar
                     * djny_s_dv_spN]

            ## b_p derivatives defined on the right boundary
            ######################################################
            djpx_sm1_defp_sm1, djpx_sm1_defp_s, djpx_sm1_dv_sm1, djpx_sm1_dv_s =\
            get_jp_derivs(efp_sm1, efp_s, v_sm1, v_s, dx_im1, params)

            djpy_s_defp_s, djpy_s_defp_spN, djpy_s_dv_s, djpy_s_dv_spN = \
            get_jp_derivs(efp_s, efp_spN, v_s, v_spN, dy_j, params)

            djpy_smN_defp_smN, djpy_smN_defp_s, djpy_smN_dv_smN, djpy_smN_dv_s = \
            get_jp_derivs(efp_smN, efp_s, v_smN, v_s, dy_jm1, params)

            rows += 9*[fp_row]
            columns += [efp_smN_col, v_smN_col, efp_sm1_col, v_sm1_col, efn_s_col,\
                        efp_s_col, v_s_col, efp_spN_col, v_spN_col]

            data += [mu_smN * dxbar/dybar * djpy_smN_defp_smN, \
                     mu_smN * dxbar/dybar * djpy_smN_dv_smN, \
                     mu_sm1 * djpx_sm1_defp_sm1, mu_sm1 * djpx_sm1_dv_sm1, \
                     -dxbar * drr_defn_s,\
                     mu_sm1 * djpx_sm1_defp_s + dxbar * (-drr_defp_s - (mu_s * djpy_s_defp_s \
                     - mu_smN * djpy_smN_defp_s) / dybar) - scp[1] * p_s, \
                     mu_sm1 * djpx_sm1_dv_s + dxbar * (-drr_dv_s - (mu_s * djpy_s_dv_s \
                     - mu_smN * djpy_smN_dv_s)/ dybar) + scp[1] * p_s, \
                     -mu_s * dxbar/dybar * djpy_s_defp_spN, -mu_s * dxbar/dybar
                     * djpy_s_dv_spN]

            ## b_v derivative
            #######################
            rows.append(fv_row)
            columns.append(v_s_col)
            data.append(1) # dv_s = 0

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

            # top_n
            ####################
            rows += [fn_row, fn_row]
            columns += [efn_s_col, efn_spN_col]
            data += [1, -1]

            # top_p
            ####################
            rows += [fp_row, fp_row]
            columns += [efp_s_col, efp_spN_col]
            data += [1, -1]

            ## bottom_v
            ####################
            rows += [fv_row, fv_row]
            columns += [v_s_col, v_spN_col]
            data += [1, -1]

    J = csr_matrix((data, (rows, columns)), shape=(3*Nx*Ny, 3*Nx*Ny))
    return vec, J
