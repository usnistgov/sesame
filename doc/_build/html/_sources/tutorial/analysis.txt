Simulation data analysis
========================

The relevant packages to import are the following::

    from sesame.utils import maps3D, extra_charges_path, get_xyz_from_s
    from sesame.observables import get_jn, get_jp, get_rr, get_n, get_p
    import matplotlib.pyplot as plt
    import numpy as np

In order to get integrated quantities, I find convenient to use an spline
interpolation. This procedure requires another routine::

    from scipy.interpolate import InterpolatedUnivariateSpline as spline

First, we load the data file for the results obtained for an applied voltage
1.0::

    efn, efp, v = np.load('data.vapp_1.0.npy')

3D map of the electrostatic potential::

    maps3D(sys, v)

Compute the current integrated across the system::

    # Define the sites between which computing the currents
    sites_i = [sys.nx//2 + j*sys.nx for j in range(sys.ny)]
    sites_ip1 = [sys.nx//2+1 + j*sys.nx for j in range(sys.ny)]
    # And the correspong lattice dimensions
    dl = sys.dx[sys.nx//2]

    # Compute the electron and hole currents
    jn = get_jn(sys, efn, v, sites_i, sites_ip1, dl)
    jp = get_jp(sys, efp, v, sites_i, sites_ip1, dl)

    # Interpolate the results and integrate over the y-direction
    j = spline(sys.ypts, jn+jp).integral(sys.ypts[0], sys.ypts[-1])

To compute the bulk recombination current we first interpolate and integrate the
recombination along the x-axis, then we do the same along the y-axis:: 

    u = []
    for j in range(sys.ny):
        # List of sites
        s = [i + j*sys.nx for i in range(sys.nx)]

        # Carrier densities
        n = get_n(sys, efn, v, s)
        p = get_p(sys, efp, v, s)

        # Recombination
        r = get_rr(sys, n, p, sys.n1[s], sys.p1[s], sys.tau_e[s], sys.tau_h[s], s)
        sp = spline(sys.xpts, r)
        u.append(sp.integral(sys.xpts[0], sys.xpts[-1]))

    sp = spline(sys.ypts, u)
    JR = sp.integral(sys.ypts[0], sys.ypts[-1])



