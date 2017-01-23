Tutorial 5: Analysis of simulation data
------------------------------------------
In this tutorial we show how to extract the data computed by the solvers. We
will use the system created in :doc:`tutorial 3 <tuto3>`.

The data analysis requires to compute carrier densities, currents and plot data.
In order to avoid having to deal with the folded discretized system, we provide
a set of methods callable with real space coordinates. These methods are
available via the :func:`~sesame.analyzer.Analyzer` object. In the code below we
load a data file and create this object::

    import numpy as np
    import sesame

    # import the system
    from jv_curve import system

    results = np.load('data.vapp_0.npz')
    sys = system()
    az = sesame.Analyzer(sys, results)

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

The descriptions of the methods available via the
:func:`~sesame.analyzer.Analyzer` object are detailed in
Sec. :ref:`label_code`. Our first example shows how to obtain integrated
quantities like the current. In the code below we compute the current for all
applied voltages of the IV curve::

    J = []
    for i in range(40):
        results = np.load('data.vapp_{0}.npz'.format(i))
        az = sesame.Analyzer(system(), results)
        J.append(az.full_current())

Non-integrated quantities are often plotted along lines. We define such lines by
two points. Given two points in real coordinates, the method
:func:`~sesame.analyzer.Analyzer.line` returns the dimensionless curvilinear
abscissae along the line, and the grid sites::

    p1 = (20e-9, 2.5e-6)   #[m]
    p2 = (2.9e-6, 2.5e-6)  #[m]

    X, sites = az.line(sys, p1, p2)

Scalar quantities like densities or recombination are obtained either for the
entire system, or on a line::

    # For the entire system
    n = az.electron_density()

    # On the previously defined line
    n = az.electron_density((p1, p2))

Data computed for the entire system are one-dimensional arrays of the folded
discretized system.

Vectorial quantities (i.e. currents) are computed either on a line or for the
entire system, by component. For instance, to compute the electron current in
the x-direction for all sites::

    # For the entire system
    jn = az.electron_current(component='x')

or on a line::

    # On the previously defined line
    jn = az.electron_current(location=(p1, p2))

We now turn to a full example that treats the line defects introduced in our
system::

    # Get the abscissae of the line defects and the corresponding sites
    p1 = (20e-9, 2.5e-6)   #[m]
    p2 = (2.9e-6, 2.5e-6)  #[m]
    X, sites = az.line(sys, p1, p2)

    # raw data
    efn = results['efn'][sites]
    efp = result['efp'][sites]
    v   = result['v'][sites]


    # Get the defect state equilibrium densities
    E = -0.25 # eV
    nGB = sys.nextra[0](sites, E)
    pGB = sys.pextra[0](sites, E)

    # Compute the carrier densities
    n = az.electron_density((p1, p2))
    p = az.hole_density((p1, p2))

    # Compute the normalized surface recombination velocity and the recombination
    S = 1e5*1e-2 / sys.scaling.velocity
    ni = sys.ni[0] # intrinsic density taken at the first site (random)
    R = S * (n*p - ni**2) / (n + nGB + p + pGB)

    # R is a 1D array containing the recombination at all the defect sites. To
    # obtain the recombination current we interpolate and integrate:
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    sp = spline(X, R)
    JGB = sp.integral(X[0], X[-1])

Observe how we accessed the dimension of the surface recombination velocity with
``sys.scaling.velocity``. Other dimensions can be obtained similarly with the
self-explanatory field names density, energy, mobility, time, length,
generation.

.. seealso:: In case the methods available in the
   :func:`~sesame.analyzer.Analyzer` are not enough (especially in 3D), the
   module :func:`sesame.observables` gives
   access to low-level routines that compute the carrier densities and the currents
   for any given sites on the discretized system.

