import numpy as np

from sesame.observables2 import *

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
    dx, dy, dz = sys.dx, sys.dy, sys.dz

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
    sites = [i + j*Nx + k*Nx*Ny for k in range(Nz) for j in range(Ny) for i in range(Nx)]

    # carrier densities
    n = get_n(sys, efn, v, sites)
    p = get_p(sys, efp, v, sites)

    # bulk charges
    rho = sys.rho - n + p

    # recombination rates
    r = get_rr(sys, n, p, sys.n1, sys.p1, sys.tau_e, sys.tau_h, sites)

    # extra charge density
    if hasattr(sys, 'Nextra'): 
        # find sites containing extra charges
        matches = sys.extra_charge_sites

        nextra = sys.nextra[matches]
        pextra = sys.pextra[matches]
        _n = n[matches]
        _p = p[matches]

        # extra charge density
        f = (_n + pextra) / (_n + _p + nextra + pextra)
        rho[matches] += sys.Nextra[matches] / 2. * (1 - 2*f)

        # extra charge recombination
        r[matches] += get_rr(sys, _n, _p, nextra, pextra, 1/sys.Sextra[matches],
                             1/sys.Sextra[matches], matches)

    # charge devided by epsilon
    rho = rho / sys.epsilon[sites]

    # Drift-diffusion equation that determines fn and fp
    def drift_diffusion(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                        dx, dxm1, dy, dym1, dz, dzm1, sites):

        # lattice distances
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        # pairs of sites in the x-direction (always defined like this)
        sm1_s = [i for i in zip(sites - 1, sites)]
        s_sp1 = [i for i in zip(sites, sites + 1)]

        # compute the currents
        jnx_s    = get_jn(sys, efn, v, s_sp1, dx)
        jnx_sm1  = get_jn(sys, efn, v, sm1_s, dxm1)
        jny_s    = get_jn(sys, efn, v, s_spN, dy)
        jny_smN  = get_jn(sys, efn, v, smN_s, dym1)
        jnz_s    = get_jn(sys, efn, v, s_spNN, dz)
        jnz_smNN = get_jn(sys, efn, v, smNN_s, dzm1)

        jpx_s    = get_jp(sys, efp, v, s_sp1, dx)
        jpx_sm1  = get_jp(sys, efp, v, sm1_s, dxm1)
        jpy_s    = get_jp(sys, efp, v, s_spN, dy)
        jpy_smN  = get_jp(sys, efp, v, smN_s, dym1)
        jpz_s    = get_jp(sys, efp, v, s_spNN, dz)
        jpz_smNN = get_jp(sys, efp, v, smNN_s, dzm1)

        fn = (jnx_s - jnx_sm1) / dxbar + (jny_s - jny_smN) / dybar \
           + (jnz_s - jnz_smNN) / dzbar + sys.g[sites] - r[sites]

        fp = (jpx_s - jpx_sm1) / dxbar + (jpy_s - jpy_smN) / dybar \
           + (jpz_s - jpz_smNN) / dzbar + r[sites] - sys.g[sites]

        return fn, fp

    # Poisson equation that determines fv
    def poisson(sys, v_smNN, v_smN, v_spN, v_spNN, dx, dxm1,
                dy, dym1, dz, dzm1, sites):

        # lattice distances
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.

        # Potentials that are always defined the same way (x-direction)
        v_sm1 = v[sites - 1]
        v_s = v[sites]
        v_sp1 = v[sites + 1]

        fv = ((v_s - v_sm1) / dxm1 - (v_sp1 - v_s) / dx) / dxbar\
           + ((v_s - v_smN) / dym1 - (v_spN - v_s) / dy) / dybar\
           + ((v_s - v_smNN) / dzm1 - (v_spNN - v_s) / dz) / dzbar\
           - rho[sites]

        return fv


    ###########################################################################
    #       inside the system: 0 < i < Nx-1, 0 < j < Ny-1, 0 < k < Nz-1       #
    ###########################################################################
    # We compute fn, fp, fv  on the inner part of the system. All the edges
    # containing boundary conditions.

    # list of the sites inside the system
    sites = [i + j*Nx + k*Nx*Ny for k in range(1,Nz-1) 
                                for j in range(1,Ny-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], (Ny-2)*(Nz-2))
    dy = np.repeat(sys.dy[1:], (Nx-2)*(Nz-2))
    dz = np.repeat(sys.dz[1:], (Nx-2)*(Ny-2))
    dxm1 = np.tile(sys.dx[:-1], (Ny-2)*(Nz-2))
    dym1 = np.repeat(sys.dy[:-1], (Nx-2)*(Nz-2))
    dzm1 = np.repeat(sys.dz[:-1], (Nx-2)*(Ny-2))

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    # compute fn, fp
    fn, fp = drift_diffusion(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                             dx, dxm1, dy, dym1, dz, dzm1, sites)

    # compute fv
    v_smNN = v[sites - Nx*Ny]
    v_smN = v[sites - Nx]
    v_spN = v[sites + Nx]
    v_spNN = v[sites + Nx*Ny]

    fv = poisson(sys, v_smNN, v_smN, v_spN, v_spNN, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update vector
    update(fn, fp, fv, sites)

    ###########################################################################
    #        left boundary: i = 0, 0 <= j <= Ny-1, 0 <= k <= Nz-1             #
    ###########################################################################
    # list of the sites on the left side
    sites = [j*Nx + k*Nx*Ny for k in range(Nz) for j in range(Ny)]
    sites = np.asarray(sites)

    # compute the currents
    s_sp1 = [sites, sites + 1]
    jnx = get_jn(sys, efn, v, s_sp1)
    jpx = get_jp(sys, efp, v, s_sp1)

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
    # We write everything that won't change in this function below

    def right_bc(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,\
                 dym1, dz, dzm1, sites):

        # lattice distances and sites
        dxbar = (dx + dxm1) / 2.
        dybar = (dy + dym1) / 2.
        dzbar = (dz + dzm1) / 2.
        sm1_s = [sites - 1, sites]

        # currents
        jnx_sm1  = get_jn(sys, efn, v, sm1_s)
        jny_s    = get_jn(sys, efn, v, s_spN, dy)
        jny_smN  = get_jn(sys, efn, v, smN_s, dym1)
        jnz_s    = get_jn(sys, efn, v, s_spNN, dz)
        jnz_smNN = get_jn(sys, efn, v, smNN_s, dzm1)

        jpx_sm1  = get_jp(sys, efp, v, sm1_s)
        jpy_s    = get_jp(sys, efp, v, s_spN, dy)
        jpy_smN  = get_jp(sys, efp, v, smN_s, dym1)
        jpz_s    = get_jp(sys, efp, v, s_spNN, dz)
        jpz_smNN = get_jp(sys, efp, v, smNN_s, dzm1)

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
    #         right boundary: i = Nx-1, 0 < j < Ny-1, 0 < k < Nz-1            #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + j*Nx + k*Nx*Ny for k in range(1,Nz-1) for j in range(1,Ny-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[-1], (Ny-2)*(Nz-2))
    dxm1 = np.tile(sys.dx[-1], (Ny-2)*(Nz-2))
    dy = np.repeat(sys.dy[1:], Nz-2)
    dym1 = np.repeat(sys.dy[:-1], Nz-2)
    dz = np.repeat(sys.dz[1:], Ny-2)
    dzm1 = np.repeat(sys.dz[:-1], Ny-2)

    # sites for the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v,  smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,\
             dym1, dz, dzm1, sites)

    ###########################################################################
    #           right boundary: i = Nx-1, j = Ny-1, 0 < k < Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + (Ny-1)*Nx + k*Nx*Ny for k in range(1,Nz-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[-1], Nz-2)
    dxm1 = np.tile(sys.dx[-1], Nz-2)
    dy = np.repeat((sys.dy[0] + sys.dy[-1]) / 2., Nz-2)
    dym1 = np.repeat(sys.dy[-1], Nz-2) 
    dz = sys.dz[1:]
    dzm1 = sys.dz[:-1]

    # compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites - Nx*(Ny-1)]
    s_spNN = [sites, sites + Nx*Ny]

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v,  smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,\
             dym1, dz, dzm1, sites)
 
    ###########################################################################
    #              right boundary: i = Nx-1, j = 0, 0 < k < Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + k*Nx*Ny for k in range(1,Nz-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = np.repeat(sys.dy[-1], Nz-2)
    dym1 =  np.repeat((sys.dy[0] + sys.dy[-1]) / 2., Nz-2)
    dz = sys.dz[1:]
    dzm1 = sys.dz[:-1]

    # compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites + Nx*(Ny-1), sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v,  smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,\
             dym1, dz, dzm1, sites)
 
    ###########################################################################
    #           right boundary: i = Nx-1, 0 < j < Ny-1, k = Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + j*Nx + (Nz-1)*Nx*Ny for j in range(1,Ny-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[-1], Ny-2)
    dxm1 = np.tile(sys.dx[-1], Ny-2)
    dy = sys.dy[1:]
    dym1 = sys.dy[:-1]
    dz = np.repeat((sys.dz[-1] + sys.dz[0])/2., Ny-2)
    dzm1 = np.repeat(sys.dz[-1], Ny-2)

    # compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites - Nx*Ny*(Nz-1)]

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v,  smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,\
             dym1, dz, dzm1, sites)

    ###########################################################################
    #              right boundary: i = Nx-1, 0 < j < Ny-1, k = 0              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + j*Nx for j in range(1,Ny-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = sys.dy[1:]
    dym1 = sys.dy[:-1]
    dz = np.repeat(sys.dz[0], Ny-2)
    dzm1 = np.repeat((sys.dz[-1] + sys.dz[0])/2., Ny-2)

    # compute the currents
    smNN_s = [sites + Nx*Ny*(Nz-1), sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v,  smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,\
             dym1, dz, dzm1, sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = Ny-1, k = 0              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + (Ny-1)*Nx]
    sites = np.asarray(sites)

    # lattice distances
    dx = sys.dx[-1]
    dxm1 = sys.dx[-1]
    dy = (sys.dy[0] + sys.dy[-1])/2.
    dym1 = sys.dy[-1]
    dz = sys.dz[0]
    dzm1 = (sys.dz[-1] + sys.dz[0])/2.

    # compute the currents
    smNN_s = [sites + Nx*Ny*(Nz-1), sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites - Nx*(Ny-1)]
    s_spNN = [sites, sites + Nx*Ny]

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v,  smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,\
             dym1, dz, dzm1, sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = Ny-1, k = Nz-1           #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + (Ny-1)*Nx + (Nz-1)*Nx*Ny]
    sites = np.asarray(sites)

    # lattice distances
    dy = (sys.dy[0] + sys.dy[-1])/2.
    dym1 = sys.dy[-1]
    dz = (sys.dz[-1] + sys.dz[0])/2.
    dzm1 = sys.dz[-1]

    # compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites - Nx*(Ny-1)]
    s_spNN = [sites, sites - Nx*Ny*(Nz-1)]

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v,  smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,\
             dym1, dz, dzm1, sites)
             
    ###########################################################################
    #                  right boundary: i = Nx-1, j = 0, k = Nz-1              #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1 + (Nz-1)*Nx*Ny]
    sites = np.asarray(sites)

    # lattice distances
    dy = sys.dy[0]
    dym1 = (sys.dy[0] + sys.dy[-1])/2.
    dz = (sys.dz[-1] + sys.dz[0])/2.
    dzm1 = sys.dz[-1]

    # compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites + Nx*(Ny-1), sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites - Nx*Ny*(Nz-1)]

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v,  smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,\
             dym1, dz, dzm1, sites)

    ###########################################################################
    #                  right boundary: i = Nx-1, j = 0, k = 0                 #
    ###########################################################################
    # list of the sites on the right side
    sites = [Nx-1]
    sites = np.asarray(sites)

    # lattice distances
    dy = sys.dy[0]
    dym1 = (sys.dy[0] + sys.dy[-1])/2.
    dz = sys.dz[0]
    dzm1 = (sys.dz[-1] + sys.dz[0])/2.

    # compute the currents
    smNN_s = [sites + Nx*Ny*(Nz-1), sites]
    smN_s = [sites + Nx*(Ny-1), sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    # compute the BC and update the right hand side vector
    right_bc(sys, efn, efp, v,  smNN_s, smN_s, s_spN, s_spNN, dx, dxm1, dy,\
             dym1, dz, dzm1, sites)



    ###########################################################################
    #            faces between contacts: 0 < i < Nx-1, j or k fixed           #
    ###########################################################################
    # Here we focus on the faces between the contacts. There are 4 cases
    # (obviously).

    ###########################################################################
    #              z-face top: 0 < i < Nx-1, 0 < j < Ny-1, k = Nz-1           #
    ###########################################################################
    # list of the sites
    sites = [i + j*Nx + (Nz-1)*Nx*Ny for j in range(1,Ny-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny-2)
    dy = np.repeat(sys.dy[1:], Nx-2)
    dz = np.repeat((sys.dz[0] + sys.dz[-1])/2., (Nx-2)*(Ny-2))
    dxm1 = np.tile(sys.dx[:-1], Ny-2)
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dzm1 = np.repeat(sys.dz[-1], (Nx-2)*(Ny-2))

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites - Nx*Ny*(Nz-1)]

    # compute fn, fp
    fn, fp = drift_diffusion(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                             dx, dxm1, dy, dym1, dz, dzm1, sites)

    # compute fv
    v_smNN = v[sites - Nx*Ny]
    v_smN = v[sites - Nx]
    v_spN = v[sites + Nx]
    v_spNN = v[sites - Nx*Ny*(Nz-1)]

    fv = poisson(sys, v_smNN, v_smN, v_spN, v_spNN, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update vector
    update(fn, fp, fv, sites)

    ###########################################################################
    #             z- face bottom: 0 < i < Nx-1, 0 < j < Ny-1, k = 0           #
    ###########################################################################
    # list of the sites
    sites = [i + j*Nx for j in range(1,Ny-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Ny-2)
    dy = np.repeat(sys.dy[1:], Nx-2)
    dz = np.repeat(sys.dz[0], (Nx-2)*(Ny-2))
    dxm1 = np.tile(sys.dx[:-1], Ny-2)
    dym1 = np.repeat(sys.dy[:-1], Nx-2)
    dzm1 = np.repeat((sys.dz[0] + sys.dz[-1])/2., (Nx-2)*(Ny-2))

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites + Nx*Ny*(Nz-1), sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    # compute fn, fp
    fn, fp = drift_diffusion(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                             dx, dxm1, dy, dym1, dz, dzm1, sites)

    # compute fv
    v_smNN = v[sites + Nx*Ny*(Nz-1)]
    v_smN = v[sites - Nx]
    v_spN = v[sites + Nx]
    v_spNN = v[sites + Nx*Ny]

    fv = poisson(sys, v_smNN, v_smN, v_spN, v_spNN, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update vector
    update(fn, fp, fv, sites)

    ###########################################################################
    #            y-face front: 0 < i < Nx-1, j = 0, 0 < k < Nz-1              #
    ###########################################################################
    # list of the sites
    sites = [i + k*Nx*Ny for k in range(1,Nz-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Nz-2)
    dy = np.repeat(sys.dy[0], (Nx-2)*(Nz-2))
    dz = np.repeat(sys.dz[1:], (Nx-2))
    dxm1 = np.tile(sys.dx[:-1], Nz-2)
    dym1 = np.repeat((sys.dy[0] + sys.dy[-1])/2., (Nx-2)*(Nz-2))
    dzm1 = np.repeat(sys.dz[:-1], Nx-2)

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites + Nx*(Ny-1), sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    # compute fn, fp
    fn, fp = drift_diffusion(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                             dx, dxm1, dy, dym1, dz, dzm1, sites)

    # compute fv
    v_smNN = v[sites - Nx*Ny]
    v_smN = v[sites + Nx*(Ny-1)]
    v_spN = v[sites + Nx]
    v_spNN = v[sites + Nx*Ny]

    fv = poisson(sys, v_smNN, v_smN, v_spN, v_spNN, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update vector
    update(fn, fp, fv, sites)

    ###########################################################################
    #            y-face back: 0 < i < Nx-1, j = Ny-1, 0 < k < Nz-1            #
    ###########################################################################
    # list of the sites
    sites = [i + (Ny-1)*Nx + k*Nx*Ny for k in range(1,Nz-1) for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dx = np.tile(sys.dx[1:], Nz-2)
    dy = np.repeat((sys.dy[0] + sys.dy[-1])/2., (Nx-2)*(Nz-2))
    dz = np.repeat(sys.dz[1:], Nx-2)
    dxm1 = np.tile(sys.dx[:-1], Nz-2)
    dym1 = np.repeat(sys.dy[0], (Nx-2)*(Nz-2))
    dzm1 = np.repeat(sys.dz[:-1], Nx-2)

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites - Nx*(Ny-1)]
    s_spNN = [sites, sites + Nx*Ny]

    # compute fn, fp
    fn, fp = drift_diffusion(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                             dx, dxm1, dy, dym1, dz, dzm1, sites)

    # compute fv
    v_smNN = v[sites - Nx*Ny]
    v_smN = v[sites - Nx]
    v_spN = v[sites - Nx*(Ny-1)]
    v_spNN = v[sites + Nx*Ny]

    fv = poisson(sys, v_smNN, v_smN, v_spN, v_spNN, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update vector
    update(fn, fp, fv, sites)


    ###########################################################################
    #           edges between contacts: 0 < i < Nx-1, j and k fixed           #
    ###########################################################################
    # Here we focus on the edges between the contacts. There are 4 cases again
    # (obviously).

    # lattice distances
    dx = sys.dx[1:]
    dxm1 = sys.dx[:-1]

    ###########################################################################
    #         edge z top // y back: 0 < i < Nx-1, j = Ny-1, k = Nz-1          #
    ###########################################################################
    # list of the sites
    sites = [i + (Ny-1)*Nx + (Nz-1)*Nx*Ny for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = np.repeat((sys.dy[0] + sys.dy[-1])/2., Nx-2)
    dz = np.repeat((sys.dz[0] + sys.dz[-1])/2., Nx-2)
    dym1 = np.repeat(sys.dy[-1], Nx-2)
    dzm1 = np.repeat(sys.dz[-1], Nx-2)

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites - Nx*(Ny-1)]
    s_spNN = [sites, sites - Nx*Ny*(Nz-1)]

    # compute fn, fp
    fn, fp = drift_diffusion(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                             dx, dxm1, dy, dym1, dz, dzm1, sites)
    # compute fv
    v_smNN = v[sites - Nx*Ny]
    v_smN = v[sites - Nx]
    v_spN = v[sites - Nx*(Ny-1)]
    v_spNN = v[sites - Nx*Ny*(Nz-1)]

    fv = poisson(sys, v_smNN, v_smN, v_spN, v_spNN, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update vector
    update(fn, fp, fv, sites)

    ###########################################################################
    #           edge z top // y front: 0 < i < Nx-1, j = 0, k = Nz-1          #
    ###########################################################################
    # list of the sites
    sites = [i + (Nz-1)*Nx*Ny for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = np.repeat(sys.dy[0], Nx-2)
    dz = np.repeat((sys.dz[0] + sys.dz[-1])/2., Nx-2)
    dym1 = np.repeat((sys.dy[0] + sys.dy[-1])/2., Nx-2)
    dzm1 = np.repeat(sys.dz[-1], Nx-2)

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites - Nx*Ny, sites]
    smN_s = [sites + Nx*(Ny-1), sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites - Nx*Ny*(Nz-1)]

    # compute fn, fp
    fn, fp = drift_diffusion(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                             dx, dxm1, dy, dym1, dz, dzm1, sites)

    # compute fv
    v_smNN = v[sites - Nx*Ny]
    v_smN = v[sites + Nx*(Ny-1)]
    v_spN = v[sites + Nx]
    v_spNN = v[sites - Nx*Ny*(Nz-1)]

    fv = poisson(sys, v_smNN, v_smN, v_spN, v_spNN, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update vector
    update(fn, fp, fv, sites)

    ###########################################################################
    #          edge z bottom // y back: 0 < i < Nx-1, j = Ny-1, k = 0         #
    ###########################################################################
    # list of the sites
    sites = [i + (Ny-1)*Nx for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = np.repeat((sys.dy[0] + sys.dy[-1])/2., Nx-2)
    dz = np.repeat(sys.dz[0], Nx-2)
    dym1 = np.repeat(sys.dy[-1], Nx-2)
    dzm1 = np.repeat((sys.dz[0] + sys.dz[-1])/2., Nx-2)

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites + Nx*Ny*(Nz-1), sites]
    smN_s = [sites - Nx, sites]
    s_spN = [sites, sites - Nx*(Ny-1)]
    s_spNN = [sites, sites + Nx*Ny]

    # compute fn, fp
    fn, fp = drift_diffusion(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                             dx, dxm1, dy, dym1, dz, dzm1, sites)

    # compute fv
    v_smNN = v[sites + Nx*Ny*(Nz-1)]
    v_smN = v[sites - Nx]
    v_spN = v[sites - Nx*(Ny-1)]
    v_spNN = v[sites + Nx*Ny]

    fv = poisson(sys, v_smNN, v_smN, v_spN, v_spNN, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update vector
    update(fn, fp, fv, sites)

    ###########################################################################
    #         edge z bottom // y front: 0 < i < Nx-1, j = 0, k = 0            #
    ###########################################################################
    # list of the sites
    sites = [i for i in range(1,Nx-1)]
    sites = np.asarray(sites)

    # lattice distances
    dy = np.repeat(sys.dy[0], Nx-2)
    dz = np.repeat(sys.dz[0], Nx-2)
    dym1 = np.repeat((sys.dy[0] + sys.dy[-1])/2., Nx-2)
    dzm1 = np.repeat((sys.dz[0] + sys.dz[-1])/2., Nx-2)

    # gather all relevant pairs of sites to compute the currents
    smNN_s = [sites + Nx*Ny*(Nz-1), sites]
    smN_s = [sites + Nx*(Ny-1), sites]
    s_spN = [sites, sites + Nx]
    s_spNN = [sites, sites + Nx*Ny]

    # compute fn, fp
    fn, fp = drift_diffusion(sys, efn, efp, v, smNN_s, smN_s, s_spN, s_spNN,\
                             dx, dxm1, dy, dym1, dz, dzm1, sites)

    # compute fv
    v_smNN = v[sites + Nx*Ny*(Nz-1)]
    v_smN = v[sites + Nx*(Ny-1)]
    v_spN = v[sites + Nx]
    v_spNN = v[sites + Nx*Ny]

    fv = poisson(sys, v_smNN, v_smN, v_spN, v_spNN, dx, dxm1, dy, dym1, dz, dzm1, sites)

    # update vector
    update(fn, fp, fv, sites)

    return vec
