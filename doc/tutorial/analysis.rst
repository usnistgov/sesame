Tutorial 4: Analysis of simulation data
------------------------------------------
In this tutorial we show how to extract the data computed by the solvers. We
will use the system created in :doc:`tutorial 3 <tuto3>`.

.. seealso:: The example treated here relies
   on the system defined in ``jv_curve.py``, which is in the
   ``examples`` directory in the root directory of the distribution. 

The data analysis requires to compute carrier densities, currents and plot data.
The relevant packages to import are the following::

    import sesame
    from sesame.observables import get_jn, get_jp, get_rr, get_n, get_p
    import numpy as np
    import matplotlib.pyplot as plt

The descriptions of these functions (input arguments, output) are detailed in
Sec. :ref:`label_code`.
In order to get integrated quantities, I find convenient to use a spline
interpolation. This procedure requires another routine::

    from scipy.interpolate import InterpolatedUnivariateSpline as spline

First we need to create the system so that we can access the discretization
easily, and the quantities used to make the system of equations dimensionless::

    from jv_curve import system
    sys = system()

In the table below we show the syntax used to get some attributes of the
:func:`~sesame.builder.Builder`

=============================               =============================================
Attribute                                   Syntax
=============================               =============================================
grid nodes                                   ``sys.xpts``, ``sys.ypts``, ``sys.zpts``
number of grid nodes                         ``sys.nx``, ``sys.ny``, ``sys.nz``
grid distances                               ``sys.dx``, ``sys.dy``, ``sys.dz``
=============================               =============================================

The exhaustive list of all accessible attributes is in the
documentation of the :func:`~sesame.builder.Builder` class itself.


Next, we load the data file for the results. As an example, let's assume we
generated a file called ``data.vapp_idx_1.npy``::

    efn, efp, v = np.load('data.vapp_idx_1.npy')

The quantities ``efn``, ``efp``, ``v`` are one-dimensional arrays that will be
used to compute and plot physical quantities. For instance, a 3D map of the
electrostatic potential of a 2D system is obtained as follows (requires
Matplotlib)::

    from sesame.plotter import map3D
    map3D(sys, v, 1e-6)


Computations of quantities like densities and currents require lists of sites
where to compute them. Remember that we folded the indices of the mesh into a single
site index ``s``

.. math:: s = i + j \times n_x + k \times n_x n_y

In 2D, the list of sites in the :math:`x`-direction between indices ``i_start``
and ``i_end``, at index ``j`` in the :math:`y`-direction reads::

    sites = [i + j*sys.nx for i in range(i_start, i_end)]

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

This is only given as an example of how to compute currents, as this particular
function is available in :func:`sesame.utils.full_current`.
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

Again, because this is very useful we implemented this function in
:func:`sesame.utils.bulk_recombination_current`.

In order to get information about the densities at the defect sites, we need to
get them. This is done by calling the function
``sesame.utils.extra_charges_path`` with the two points defining the line
defects we are considering. As an example, let's compute the recombination
current along the grain boundary::

    from sesame.utils import extra_charges_path

    # Get the defect sites, path along the lattice, x indices, y indices
    p1 = (20e-9, 2.5e-6, 0)   #[m]
    p2 = (2.9e-6, 2.5e-6, 0)  #[m]
    GBsites, X, xGB, yGB = extra_charges_path(sys, startGB, endGB)

    # Get the defect state equilibrium densities
    nGB = sys.nextra[0, GBsites]
    pGB = sys.pextra[0, GBsites]

    # Compute the carrier densities
    n = get_n(sys, efn, v, GBsites)
    p = get_p(sys, efp, v, GBsites)

    # Compute the normalized surface recombination velocity and the recombination
    S = 1e5*1e-2 / sys.scaling.velocity
    ni = sys.ni[0] # intrinsic density taken at the first site (random)
    R = S * (n*p - ni**2) / (n + nGB + p + pGB)

    # R is an 1D array containing the recombination at all the defect sites. To
    # obtain the recombination current we interpolate and integrate:
    sp = spline(X, R)
    JGB = sp.integral(X[0], X[-1])

Observe how we accessed the dimension of the surface recombination velocity with
``sys.scaling.velocity``. Other dimensions can be obtained similarly with the
self-explanatory field names density, energy, mobility, time, length,
generation.

Once the defect sites are known, the raw data at these sites are accessible
via::

    efn = efn[GBsites]
    efp = efp[GBsites]
    v   = v[GBsites]

and can be plotted following the curvilinear abscissa of the defect::

    plt.plot(X, efn)
    plt.show()

The electron and hole currents along the line defects are computed as follows (not
computed current for the last site)::

    jn = get_jn(sys, efn, v, GBsites[:-1], GBsites[1:], X[1:]-X[:-1])
    jp = get_jp(sys, efp, v, GBsites[:-1], GBsites[1:], X[1:]-X[:-1])
