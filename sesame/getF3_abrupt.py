import numpy as np
from .observables import *

def getF(sys, v, efn, efp):
    ###########################################################################
    #               organization of the right hand side vector                #
    ###########################################################################
    # A site with coordinates (i,j,k) corresponds to a site number s as follows:
    # k = s//(Nx*Ny)
    # j = s - s//Nx
    # i = s - j*Nx - k*Nx*Ny
    #
    # Rows for (efn_s, efp_s, v_s)
    # ----------------------------
    # fn_row = 3*s
    # fp_row = 3*s+1
    # fv_row = 3*s+2

    Nx, Ny, Nz = sys.xpts.shape[0], sys.ypts.shape[0], sys.zpts.shape[0]

    # right hand side vector
    global vec
    vec = np.zeros((3*Nx*Ny*Nz,))
    def update(fn, fp, fv, sites):
        global vec
        vec[3*sites] = fn
        vec[3*sites+1] = fp
        vec[3*sites+2] = fv


    ###########################################################################
    #                     For all sites in the system                         #
    ###########################################################################
    _sites = np.arange(Nx*Ny*Nz, dtype=int)

    # carrier densities
    n = get_n(sys, efn, v, _sites)
    p = get_p(sys, efp, v, _sites)

    # bulk charges
    rho = sys.rho - n + p

    # recombination rates
    r = get_rr(sys, n, p, sys.n1, sys.p1, sys.tau_e, sys.tau_h, _sites)

    # extra charge density
    if hasattr(sys, 'Nextra'): 
        # find sites containing extra charges
        for idx, matches in enumerate(sys.extra_charge_sites):
            nextra = sys.nextra[idx, matches]
            pextra = sys.pextra[idx, matches]
            _n = n[matches]
            _p = p[matches]

            # extra charge density
            Se = sys.Seextra[idx, matches]
            Sh = sys.Shextra[idx, matches]
            f = (Se*_n + Sh*pextra) / (Se*(_n+nextra) + Sh*(_p+pextra))
            rho[matches] += sys.Nextra[idx, matches] / 2. * (1 - 2*f)

            # extra charge recombination
            r[matches] += get_rr(sys, _n, _p, nextra, pextra, 1/Se, 1/Sh, matches)

    # charge devided by epsilon
    rho = rho / sys.epsilon[_sites]

    # reshape the array as array[z-indices, y-indices, x-indices]
    _sites = _sites.reshape(Nz, Ny, Nx)

    def currents(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites):
        jnx_s, jnx_sm1, jny_s, jny_smN, jnz_s, jnz_smNN = 0, 0, 0, 0, 0, 0
        jpx_s, jpx_sm1, jpy_s, jpy_smN, jpz_s, jpz_smNN = 0, 0, 0, 0, 0, 0

        if dx.all() != 0:
            jnx_s = get_jn(sys, efn, v, sites, sites + 1, dx)
            jpx_s = get_jp(sys, efp, v, sites, sites + 1, dx)
        if dxm1.all() != 0:
            jnx_sm1 = get_jn(sys, efn, v, sites - 1, sites, dxm1)
            jpx_sm1 = get_jp(sys, efp, v, sites - 1, sites, dxm1)
        if dy.all() != 0:
            jny_s = get_jn(sys, efn, v, sites, sites + Nx, dy)
            jpy_s = get_jp(sys, efp, v, sites, sites + Nx, dy)
        if dym1.all() != 0:
            jny_smN = get_jn(sys, efn, v, sites - Nx, sites, dym1)
            jpy_smN = get_jp(sys, efp, v, sites - Nx, sites, dym1)
        if dz.all() != 0:
            jnz_s = get_jn(sys, efn, v, sites, sites + Nx*Ny, dz)
            jpz_s = get_jp(sys, efp, v, sites, sites + Nx*Ny, dz)
        if dzm1.all() != 0:
            jnz_smNN = get_jn(sys, efn, v, sites - Nx*Ny, sites, dzm1)
            jpz_smNN = get_jp(sys, efp, v, sites - Nx*Ny, sites, dzm1)

        return jnx_s, jnx_sm1, jny_s, jny_smN, jnz_s, jnz_smNN,\
               jpx_s, jpx_sm1, jpy_s, jpy_smN, jpz_s, jpz_smNN 

    def ddp(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites):
    # Drift diffusion Poisson equations that determine fn, fp, fv

        # lattice distances
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        # compute currents
        jnx_s, jnx_sm1, jny_s, jny_smN, jnz_s, jnz_smNN,\
        jpx_s, jpx_sm1, jpy_s, jpy_smN, jpz_s, jpz_smNN = \
        currents(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

        # drift diffusion
        u = sys.g[sites] - r[sites]
        fn = (jnx_s - jnx_sm1) / dxbar + (jny_s - jny_smN) / dybar \
           + (jnz_s - jnz_smNN) / dzbar + u
        fp = (jpx_s - jpx_sm1) / dxbar + (jpy_s - jpy_smN) / dybar \
           + (jpz_s - jpz_smNN) / dzbar - u
          
        # Poisson
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

        # update vector
        update(fn, fp, fv, sites)

    def right_bc(sys, efn, efp, v, dy, dym1, dz, dzm1, sites):
    # Boundary conditions on the right contact

        # lattice distances and sites
        dx = np.array([0])
        dxm1 = sys.dx[-1]
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        # compute currents
        _, jnx_sm1, jny_s, jny_smN, jnz_s, jnz_smNN,\
        _, jpx_sm1, jpy_s, jpy_smN, jpz_s, jpz_smNN = \
        currents(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

        # compute jx_s with continuity equation
        jnx_s = jnx_sm1 + dxbar * (r[sites] - sys.g[sites] - (jny_s - jny_smN)/dybar\
                                   - (jnz_s - jnz_smNN)/dzbar)
        jpx_s = jpx_sm1 + dxbar * (sys.g[sites] - r[sites] - (jpy_s - jpy_smN)/dybar\
                                   - (jpz_s - jpz_smNN)/dzbar)

        # b_n, b_p and b_v values
        n_eq = 0
        p_eq = 0
        if sys.rho[2*Nx-1] < 0: # p doped
            p_eq = -sys.rho[2*Nx-1]
            n_eq = sys.ni[sites]**2 / p_eq
        else: # n doped
            n_eq = sys.rho[2*Nx-1]
            p_eq = sys.ni[sites]**2 / n_eq
            
        bn = jnx_s + sys.Scn[1] * (n[sites] - n_eq)
        bp = jpx_s - sys.Scp[1] * (p[sites] - p_eq)
        bv = 0 # Dirichlet BC
        # update right hand side vector
        update(bn, bp, bv, sites)


    ###########################################################################
    #       inside the system: 0 < i < Nx-1, 0 < j < Ny-1, 0 < k < Nz-1       #
    ###########################################################################
    # We compute fn, fp, fv  on the inner part of the system.

    # list of the sites inside the system
    sites = _sites[1:Nz-1, 1:Ny-1, 1:Nx-1].flatten()

    # lattice distances
    dx = np.tile(sys.dx[1:], (Ny-2)*(Nz-2))
    dy = np.repeat(sys.dy[1:], (Nx-2)*(Nz-2))
    dz = np.repeat(sys.dz[1:], (Nx-2)*(Ny-2))
    dxm1 = np.tile(sys.dx[:-1], (Ny-2)*(Nz-2))
    dym1 = np.repeat(sys.dy[:-1], (Nx-2)*(Nz-2))
    dzm1 = np.repeat(sys.dz[:-1], (Nx-2)*(Ny-2))

    # compute fn, fp, fv and update vector
    ddp(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)


    ###########################################################################
    #        left boundary: i = 0, 0 <= j <= Ny-1, 0 <= k <= Nz-1             #
    ###########################################################################
    # list of the sites on the left side
    sites = _sites[:, :, 0].flatten()

    # compute the currents
    jnx = get_jn(sys, efn, v, sites, sites + 1, sys.dx[0])
    jpx = get_jp(sys, efp, v, sites, sites + 1, sys.dx[0])

    # compute an, ap, av
    n_eq = 0
    p_eq = 0
    #TODO tricky here to decide
    if sys.rho[Nx] < 0: # p doped
        p_eq = -sys.rho[sites]
        n_eq = sys.ni[sites]**2 / p_eq
    else: # n doped
        n_eq = sys.rho[sites]
        p_eq = sys.ni[sites]**2 / n_eq
        
    an = jnx - sys.Scn[0] * (n[sites] - n_eq)
    ap = jpx + sys.Scp[0] * (p[sites] - p_eq)
    av = 0 # to ensure Dirichlet BCs

    update(an, ap, av, sites)

    ###########################################################################
    #                            right boundaries                             #
    ###########################################################################

    ###########################################################################
    #         right boundary: i = Nx-1, 0 < j < Ny-1, 0 < k < Nz-1            #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[1:Nz-1, 1:Ny-1, Nx-1].flatten()

    # lattice distances
    dy = np.repeat(sys.dy[1:], Nz-2)
    dym1 = np.repeat(sys.dy[:-1], Nz-2)
    dz = np.repeat(sys.dz[1:], Ny-2)
    dzm1 = np.repeat(sys.dz[:-1], Ny-2)

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #           right boundary: i = Nx-1, j = Ny-1, 0 < k < Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[1:Nz-1, Ny-1, Nx-1].flatten()

    # lattice distances
    dy = np.array([0])
    dym1 = np.repeat(sys.dy[-1], Nz-2) 
    dz = sys.dz[1:]
    dzm1 = sys.dz[:-1]

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)
 
    ###########################################################################
    #              right boundary: i = Nx-1, j = 0, 0 < k < Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[1:Nz-1, 0, Nx-1].flatten()

    # lattice distances
    dy = np.repeat(sys.dy[-1], Nz-2)
    dym1 =  np.array([0])
    dz = sys.dz[1:]
    dzm1 = sys.dz[:-1]

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)
 
    ###########################################################################
    #           right boundary: i = Nx-1, 0 < j < Ny-1, k = Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[Nz-1, 1:Ny-1, Nx-1].flatten()

    # lattice distances
    dy = sys.dy[1:]
    dym1 = sys.dy[:-1]
    dz = np.array([0])
    dzm1 = np.repeat(sys.dz[-1], Ny-2)

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #              right boundary: i = Nx-1, 0 < j < Ny-1, k = 0              #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[0, 1:Ny-1, Nx-1].flatten()

    # lattice distances
    dy = sys.dy[1:]
    dym1 = sys.dy[:-1]
    dz = np.repeat(sys.dz[0], Ny-2)
    dzm1 = np.array([0])

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = Ny-1, k = 0              #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[0, Ny-1, Nx-1].flatten()

    # lattice distances
    dy = np.array([0])
    dym1 = sys.dy[-1]
    dz = sys.dz[0]
    dzm1 = np.array([0])

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = Ny-1, k = Nz-1           #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[Nz-1, Ny-1, Nx-1].flatten()

    # lattice distances
    dy = np.array([0])
    dym1 = sys.dy[-1]
    dz = np.array([0])
    dzm1 = sys.dz[-1]

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)
             
    ###########################################################################
    #                  right boundary: i = Nx-1, j = 0, k = Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[Nz-1, 0, Nx-1].flatten()

    # lattice distances
    dy = sys.dy[0]
    dym1 = np.array([0])
    dz = np.array([0])
    dzm1 = sys.dz[-1]

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = 0, k = 0                 #
    ###########################################################################
    # list of the sites on the right side
    sites = _sites[0, 0, Nx-1].flatten()

    # lattice distances
    dy = sys.dy[0]
    dym1 = np.array([0])
    dz = sys.dz[0]
    dzm1 = np.array([0])

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v, dy, dym1, dz, dzm1, sites)


    ###########################################################################
    #            faces between contacts: 0 < i < Nx-1, j or k fixed           #
    ###########################################################################
    # Here we focus on the faces between the contacts.

    ###########################################################################
    #              z-face top: 0 < i < Nx-1, 0 < j < Ny-1, k = Nz-1           #
    ###########################################################################
    # list of the sites
    sites = _sites[Nz-1, 1:Ny-1, 1:Nx-1].flatten()

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny-2)
    dy = np.repeat(sys.dy[1:], Nx-2)
    dz = np.array([0])
    dxm1 = np.tile(sys.dx[:-1], Ny-2)
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dzm1 = np.repeat(sys.dz[-1], (Nx-2)*(Ny-2))

    # compute fn, fp, fv and update vector
    ddp(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #             z- face bottom: 0 < i < Nx-1, 0 < j < Ny-1, k = 0           #
    ###########################################################################
    # list of the sites
    sites = _sites[0, 1:Ny-1, 1:Nx-1].flatten()

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny-2)
    dy = np.repeat(sys.dy[1:], Nx-2)
    dz = np.repeat(sys.dz[0], (Nx-2)*(Ny-2))
    dxm1 = np.tile(sys.dx[:-1], Ny-2)
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dzm1 = np.array([0])

    # compute fn, fp, fv and update vector
    ddp(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #            y-face front: 0 < i < Nx-1, j = 0, 0 < k < Nz-1              #
    ###########################################################################
    # list of the sites
    sites = _sites[1:Nz-1, 0, 1:Nx-1].flatten()

    # lattice distances
    dx = np.tile(sys.dx[1:], Nz-2)
    dy = np.repeat(sys.dy[0], (Nx-2)*(Nz-2))
    dz = np.repeat(sys.dz[1:], (Nx-2))
    dxm1 = np.tile(sys.dx[:-1], Nz-2)
    dym1 = np.array([0])
    dzm1 = np.repeat(sys.dz[:-1], Nx-2)

    # compute fn, fp, fv and update vector
    ddp(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #            y-face back: 0 < i < Nx-1, j = Ny-1, 0 < k < Nz-1            #
    ###########################################################################
    # list of the sites
    sites = _sites[1:Nz-1, Ny-1, 1:Nx-1].flatten()

    # lattice distances
    dx = np.tile(sys.dx[1:], Nz-2)
    dy = np.array([0])
    dz = np.repeat(sys.dz[1:], Nx-2)
    dxm1 = np.tile(sys.dx[:-1], Nz-2)
    dym1 = np.repeat(sys.dy[0], (Nx-2)*(Nz-2))
    dzm1 = np.repeat(sys.dz[:-1], Nx-2)

    # compute fn, fp, fv and update vector
    ddp(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #           edges between contacts: 0 < i < Nx-1, j and k fixed           #
    ###########################################################################
    # Here we focus on the edges between the contacts.

    # lattice distances
    dx = sys.dx[1:]
    dxm1 = sys.dx[:-1]

    ###########################################################################
    #         edge z top // y back: 0 < i < Nx-1, j = Ny-1, k = Nz-1          #
    ###########################################################################
    # list of the sites
    sites = _sites[Nz-1, Ny-1, 1:Nx-1].flatten()

    # lattice distances
    dy = np.array([0])
    dz = np.array([0])
    dym1 = np.repeat(sys.dy[-1], Nx-2)
    dzm1 = np.repeat(sys.dz[-1], Nx-2)

    # compute fn, fp, fv and update vector
    ddp(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #           edge z top // y front: 0 < i < Nx-1, j = 0, k = Nz-1          #
    ###########################################################################
    # list of the sites
    sites = _sites[Nz-1, 0, 1:Nx-1].flatten()

    # lattice distances
    dy = np.repeat(sys.dy[0], Nx-2)
    dz = np.array([0])
    dym1 = np.array([0])
    dzm1 = np.repeat(sys.dz[-1], Nx-2)

    # compute fn, fp, fv and update vector
    ddp(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #          edge z bottom // y back: 0 < i < Nx-1, j = Ny-1, k = 0         #
    ###########################################################################
    # list of the sites
    sites = _sites[0, Ny-1, 1:Nx-1].flatten()

    # lattice distances
    dy = np.array([0])
    dz = np.repeat(sys.dz[0], Nx-2)
    dym1 = np.repeat(sys.dy[-1], Nx-2)
    dzm1 = np.array([0])

    # compute fn, fp, fv and update vector
    ddp(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    ###########################################################################
    #         edge z bottom // y front: 0 < i < Nx-1, j = 0, k = 0            #
    ###########################################################################
    # list of the sites
    sites = _sites[0, 0, 1:Nx-1].flatten()

    # lattice distances
    dy = np.repeat(sys.dy[0], Nx-2)
    dz = np.repeat(sys.dz[0], Nx-2)
    dym1 = np.array([0])
    dzm1 = np.array([0])

    # compute fn, fp, fv and update vector
    ddp(sys, efn, efp, v, dx, dxm1, dy, dym1, dz, dzm1, sites)

    return vec
