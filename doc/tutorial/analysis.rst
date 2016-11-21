Simulation data analysis
========================

The data analysis requires to compute carrier densities, currents and plot data.
The relevant packages to import are the following::

    from sesame.utils import maps3D, extra_charges_path
    from sesame.observables import get_jn, get_jp, get_rr, get_n, get_p
    import matplotlib.pyplot as plt
    import numpy as np

The descriptions of these functions (input arguments, output) are detailed in
Sec. :ref:`label_code`.
In order to get integrated quantities, I find convenient to use a spline
interpolation. This procedure requires another routine::

    from scipy.interpolate import InterpolatedUnivariateSpline as spline

First, we load the data file for the results obtained for an applied voltage
1.0::

    efn, efp, v = np.load('data.vapp_1.0.npy')

As an example one can plot a 3D map of the electrostatic potential::

    maps3D(sys, v)

Computing the current integrated across the system is done as follows::

    # Define the sites between which computing the currents
    sites_i = [sys.nx//2 + j*sys.nx for j in range(sys.ny)]
    sites_ip1 = [sys.nx//2+1 + j*sys.nx for j in range(sys.ny)]
    # And the corresponding lattice dimensions
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

In order to get information about the defect sites we need to get them. This is
done by calling the function ``sesame.utils.extra_charges_path`` with the
starting and ending point of the defect line we are considering. Let's compute
the recombination along the grain boundary::

    # Get the defect sites, path along the lattice, x coordinates, y coordinates
    startGB = (20e-9, 2.5e-6, 0)   #[m]
    endGB   = (2.8e-6, 2.5e-6, 0)  #[m]
    GBsites, X, xGB, yGB = extra_charges_path(sys, startGB, endGB)

    # Get the defect state equilibrium densities
    nGB = sys.nextra[0, GBsites]
    pGB = sys.pextra[0, GBsites]

    # Compute the carrier densities
    n = get_n(sys, efn, v, GBsites)
    p = get_p(sys, efp, v, GBsites)

    # Compute the normalized surface recombination velocity and the recombination
    S = 1e5*1e-2 / sys.Sc
    ni = sys.ni[0] # intrinsic density taken at the first site (random)
    R = S * (n*p - ni**2) / (n + nGB + p + pGB)

    # R is an 1D array containing the recombination at all the defect sites. To
    # obtain the recombination current we interpolate and integrate:
    sp = spline(X, R)
    JGB = sp.integral(X[0], X[-1])


Once the defect sites are known, the raw data at these sites are accessible
via::

    efn = efn[GBsites]
    efp = efp[GBsites]
    v   = v[GBsites]

and can be plotted following the curvilinear abscissa of the defect::

    plt.plot(X, efn)
    plt.show()

The electron and hole currents along the defect line are computed as follows (not
computing current for the last site)::

    jn = get_jn(sys, efn, v, GBsites[:-1], GBsites[1:], X[1:]-X[:-1])
    jp = get_jp(sys, efp, v, GBsites[:-1], GBsites[1:], X[1:]-X[:-1])
