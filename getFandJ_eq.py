import numpy as np
from numpy import exp
from scipy.sparse import csr_matrix

# remember that efn and efp are zero at equilibrium

def getFandJ_eq(v, params):
    bl = params.bl
    eg = params.eg
    xpts = params.xpts
    ypts = params.ypts
    rho = params.rho

    dx = xpts[1:] - xpts[:-1]
    dy = ypts[1:] - ypts[:-1]
    Nx = xpts.shape[0]
    Ny = ypts.shape[0]

    f = np.zeros((Nx*Ny,), dtype=float)

    rows = []
    columns = []
    data = []

    for s in range(Nx*Ny):
        j = s//Nx
        i = s - j*Nx

        fv_row = s 
        v_s_col = s
        v_sp1_col = s+1
        v_sm1_col = s-1
        v_spN_col = s + Nx
        v_smN_col = s - Nx

        ## inside the grid
        if 0 < i < Nx-1 and 0 < j < Ny-1:
            dx_i = dx[i]
            dx_im1 = dx[i-1]
            dxbar = (dx_i + dx_im1)/2.
            dy_j = dy[j]
            dy_jm1 = dy[j-1]
            dybar = (dy_j + dy_jm1)/2.

            v_s = v[s]
            v_sm1 = v[s-1]
            v_sp1 = v[s+1]
            v_smN = v[s-Nx]
            v_spN = v[s+Nx]

            fv = 1./dxbar * ((v_s-v_sm1)/dx_im1 - (v_sp1-v_s)/dx_i)\
                 + 1./dybar * ((v_s-v_smN)/dy_jm1 - (v_spN-v_s)/dy_j)\
                 - (rho[s] + exp(bl-eg-v_s) - exp(-bl+v_s))
            
            ## fv derivatives
            dfv_dvmN = -1./(dy_jm1 * dybar )
            dfv_dvm1 = -1./(dx_im1 * dxbar)
            dfv_dv = 2./(dx_i * dx_im1) + 2./(dy_j * dy_jm1) + exp(bl-eg-v_s)+exp(-bl+v_s)
            dfv_dvp1 = -1./(dx_i * dxbar)
            dfv_dvpN = -1./(dy_j * dybar)

            rows += 5*[fv_row]
            columns += [v_smN_col, v_sm1_col, v_s_col, v_sp1_col, v_spN_col]
            data += [dfv_dvmN, dfv_dvm1, dfv_dv, dfv_dvp1, dfv_dvpN]
            f[fv_row] = fv

        ## boundary conditions
        elif i == 0 or i == Nx-1: # left and right side
            # I want dv = 0 on the left and right boundaries
            rows.append(s)
            columns.append(s)
            data.append(1)
            f[fv_row] = 0

        elif j == 0: #bottom
            # I want dv_s = dv_spN
            rows += [s, s]
            columns += [s, s + Nx]
            data += [1, -1]
            f[fv_row] = 0

        elif j == Ny-1: # top
            # I want dv_s = dv_smN
            rows += [s, s]
            columns += [s, s - Nx]
            data += [1, -1]
            f[fv_row] = 0

    J = csr_matrix((data, (rows, columns)), shape=(Nx*Ny, Nx*Ny))
    return f, J
