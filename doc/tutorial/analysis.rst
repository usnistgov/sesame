Tutorial 5: Saving, loading, and analyzing simulation data
-------------------------------------------------------------

In this tutorial we describe the input and output data formats of Sesame, and show how to use Sesame's built-in tools to analyze the solution.

Saving and Loading data
^^^^^^^^^^^^^^^^^^^^^^^^

Sesame's ``save_sim`` command saves both the system object, which contains all of the simulation settings, and the solution dictionary.  An example of its use is shown below::


	sesame.save_sim(sys, results, "my_sim")

The saved output file is named "my_sim.gzip" extension.  The gzip extension indicates the data is compressed, and the data structures are stored using python's ``pickle`` module.  


The data can also be saved in a Matlab-readable format (.mat file), by adding fmt='mat' as an additional input argument:: 

	sesame.save_sim(system, result, "my_sim", fmt='mat')

In this case the arrays defining the system properties (including Eg, Nc, Nv, etc) are saved in a ``system`` data structure, and the solution (:math:`E_{F_n},E_{F_p},V`) is saved in a ``results`` data structure

Make a note of the folded array shape of the data.

Loading a saved simulation is accomplished with the command::

	sys, result = sesame.load_sim("my_sim")

Analysis of data with the Analyzer object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Next we show how to extract and analyze the data computed by the solvers. We
will use the system created in :doc:`tutorial 3 <tuto3>`, so we start by
importing the function that builds it after importing numpy and sesame::

    import numpy as np
    import sesame
   

Our data analysis will begin with computing carrier densities, currents and plotting data.
In order to avoid having to deal with the folded discretized system, we provide
a set of methods callable with real (continuous) space coordinates. In the
code below we load a data file and create an instance of this class::

    results = np.load('2dpnIV.vapp_0.npz')
    az = sesame.Analyzer(syst, results)

The ``Analyzer`` object is initialized with a system and a dictionary of
results.  This dictionary must contain the key ``v``, and can include ``efn``,
``efp`` when computed.

A summary and the descriptions of the methods available via the
:func:`~sesame.analyzer.Analyzer` object are detailed in
Sec. :ref:`label_code`.

Computing densities, recombination and currents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We start with how to obtain integrated
quantities like the steady stated current. In the code below we compute the
current for all applied voltages of the IV curve::

    J = []
    for i in range(40):
        results = np.load('2dpnIV.vapp_{0}.npz'.format(i))
        az = sesame.Analyzer(system(), results)
        J.append(az.full_current())

Non-integrated quantities are often plotted along lines. We define such lines by
two points. Given two points in real coordinates, the method
:func:`~sesame.analyzer.Analyzer.line` returns the dimensionless curvilinear
abscissae along the line, and the grid sites::

    p1 = (20e-9, 2.5e-6)   #[m]
    p2 = (2.9e-6, 2.5e-6)  #[m]

    X, sites = az.line(syst, p1, p2)

Note that the ``line`` method can be called without an instance of the
``Analyzer`` class. Just use ``sesame.Analyzer.line(syst, p1, p2)`` to get the
abscissae along a line and the sites indices.

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
the x-direction::

    # For the entire system
    jn = az.electron_current(component='x')

    # On the previously defined line
    jn = az.electron_current(location=(p1, p2))

Once these quantities are obtained, they can be plotted with ``matplotlib``, or
written to a file and plotted using any external viewer. To make the
visualization of two- and three-dimensional plots easy, ``sesame`` provides a
few functions (requiring ``matplotblib``) that represent quantities in 2D or
3D. For example, one can visualize the electrostatic potential at zero bias in
3D with::

    results = np.load('2dpnIV.vapp_0.npz')
    az = sesame.Analyzer(syst, results)
    az.map3D(results['v']) # units of kT/q

.. image:: analysis_potential.*
   :align: center

or plot the electron current accross the system::

    results = np.load('2dpnIV.vapp_10.npz')
    az = sesame.Analyzer(syst, results)
    az.electron_current_map()

.. image:: analysis_currents.*
   :align: center

We now turn to the treatment of the line defects introduced in our system::

    # Get the abscissae of the line defects and the corresponding sites
    p1 = (20e-9, 2.5e-6)   #[m]
    p2 = (2.9e-6, 2.5e-6)  #[m]
    X, sites = az.line(syst, p1, p2)

    # raw data
    efn = results['efn'][sites]
    efp = result['efp'][sites]
    v   = result['v'][sites]

    # Units of physical quantities for our system
    scaling = syst.scaling

    # Get the defect state equilibrium densities
    vt = scaling.energy
    E = -0.25 # eV
    nd = syst.ni[sites] * np.exp(+ E/vt)
    pd = syst.ni[sites] * np.exp(- E/vt)

    # Compute the carrier densities the line defect
    n = az.electron_density((p1, p2))
    p = az.hole_density((p1, p2))

    # Compute the defect recombination rate
    defect = syst.defects_list[0]
    R = az.defect_rr(defect)

    # Compute the integrated recombination along the line defect
    J = az.integrated_defect_recombination(defect)

Observe how we accessed the dimensions of physical quantities (and the energy
scale). Available dimensions are: density, energy, mobility, time, length, and
generation. These dimensions (except mobility) depend on the temperature and the
unit length (meter or centimeter) given when creating an instance of the class
:func:`~sesame.builder.Builder` (default is 300 K and centimeters).

The attribute of Builder called ``defects_list`` is a list of named tuples. This
list stores the parameters of each defect originally added to the system. The
field names of the named tuples are ``sites``, ``location``, ``dos``,
``energy``, ``sigma_e``, ``sigma_h``, ``transition``, ``perp_dl``. The last
field contains the lattice distance perpendicular to the line of defects. It is
necessary to normalize the recombination velocity and the density of states.


Advanced possibilities
^^^^^^^^^^^^^^^^^^^^^^

In case the methods available in the :func:`~sesame.analyzer.Analyzer` are not
enough (especially in 3D), the module :func:`sesame.observables` gives access to
low-level routines that compute the carrier densities and the currents for any
given sites on the discretized system.

In the table below we show the syntax used to get some attributes of the
:func:`~sesame.builder.Builder` that can then be useful:

=============================    =============================================
Attribute                        Syntax
=============================    =============================================
grid nodes                        ``syst.xpts``, ``syst.ypts``, ``syst.zpts``
number of grid nodes              ``syst.nx``, ``syst.ny``, ``syst.nz``
grid distances                    ``syst.dx``, ``syst.dy``, ``syst.dz``
=============================    =============================================

The exhaustive list of all accessible attributes is in the
documentation of the :func:`~sesame.builder.Builder` class itself. Note that the
grid nodes are in the units given in the system definition, while the grid
distances are dimensionless.
