Tutorial 1: IV curve of a one-dimensional pn junction
------------------------------------------------------

In this tutorial we show how to build a one-dimensional pn
junction and compute a IV curve.

.. seealso:: The example treated here is in the file ``1dpn.py`` in the
   ``examples`` directory in the root directory of the distribution. 

A word for Matlab users
........................
Sesame uses the Python3 language and the scientific libraries Numpy and Scipy. 
A documentation on the similarities and differences between Matlab and
Numpy/Scipy can be found `here
<https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html>`_.
   

Building a system
...................
We start by importing the sesame package and numpy::

    import sesame
    import numpy as np

A crucial point is the creation of a mesh of our system. A mesh too coarse will
give inaccurate results, and a mesh to fine will make the simulation slow. Also,
because of the strong variations of potential and densities throughout the
system, we need an irregular grid. After the tutorials the user should have a
sense of what makes an appropriate grid. In this example, we create a mesh which
contains more nodes in the pn junction depletion region::

    L = 3e-6 # length of the system in the x-direction [m]
    x = np.concatenate((np.linspace(0,1.2e-6, 100, endpoint=False), 
                        np.linspace(1.2e-6, L, 50)))

To make a system we need to create an instance of the
:func:`~sesame.builder.Builder`::

    sys = sesame.Builder(x, input_length='m')

Note that  we accessed :func:`~sesame.builder.Builder` by the name
``sesame.Builder``. We could have written ``sesame.builder.Builder`` instead.
For convenience some often used members of the sub-packages of Sesame are
accessible through the top-level `sesame` package. See the :doc:`reference
documentation <../reference/index>`.

Now we need to add a material to our system. A material is defined using a
dictionary that is then added to the system::

    CdTe = {'Nc':8e17*1e6, 'Nv':1.8e19*1e6, 'Eg':1.5, 'epsilon':9.4,
            'mu_e':100*1e-4, 'mu_h':100*1e-4, 'tau_e':10e-9, 'tau_h':10e-9}

    sys.add_material(CdTe)

where ``Nc`` (``Nv``) is the effective density of states of the conduction
(valence) band (:math:`\mathrm{m^{-3}}`), ``Eg`` is the material band gap
(:math:`\mathrm{eV}`), ``epsilon`` is the material's dielectric constant,
``mu_e`` (``mu_h``) is the electron (hole) mobility (:math:`\mathrm{m^2/(V\cdot
s)}`), ``tau_e`` (``tau_h``) is the electron (hole) bulk lifetime. For the full list
of material parameters available, see the documentation of the method
:func:`~sesame.builder.Builder.add_material` of the :func:`~sesame.builder.Builder`.


.. note::
   We assumed that a single material/region makes the entire system.
   Different regions can be specified and we show how in the addition of dopants
   below.

.. warning::
   The code does not handle regions with different band
   structures because we did not implement the equations necessary to treat the
   interfaces between them. However, different regions can have different
   mobilities or bulk lifetimes for example. More on this below and  in
   :doc:`tutorial 2 <tuto2>`.

Let's add the dopants to define a pn junction. To do this, we need to define the
regions containing each type of dopants. A region is defined by a function::

    junction = 50e-9 # extent of the junction from the left contact [m]

    def region(pos):
        x = pos
        return x < junction

The function ``region`` takes a single argument ``pos``, a tuple containing
coordinates in real space, and returns ``True`` (``False``) if this  position is
on the left (right) of the junction. The doping will be n-type for
:math:`x<junction` and p-type for :math:`x>junction`::

    # Add the donors
    nD = 1e17 * 1e6 # [m^-3]
    sys.add_donor(nD, region)

    # Add the acceptors
    region2 = lambda pos: 1 - region(pos)
    nA = 1e15 * 1e6 # [m^-3]
    sys.add_acceptor(nA, region2)

Note that we defined ``region2`` with an inline function with the keyword
``lambda``. This does the same thing as the function definition used for
``region``.  Now that we have the interior of the system, we specify the
contacts boundary conditions. We choose to have perfectly Ohmic contacts, which
are perfectly selective out of equilibrium::

    # Define Ohmic contacts
    sys.contact_type('Ohmic', 'Ohmic')

    # Define the surface recombination velocities for electrons and holes [m/s]
    Sn_left, Sp_left, Sn_right, Sp_right = 1e50, 0, 0, 1e50
    sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

If we want to make a IV curve, we need a generation profile. This is defined
as follows::

    phi = 1e21 # photon flux [1/(m^2 s)]
    alpha = 2.3e6 # absorption coefficient [1/m]

    # Define a function for the generation rate
    f = lambda x: phi * alpha * np.exp(-alpha * x)
    sys.generation(f)

We can now use this system to solve the Poisson equation at thermal equilibrium
and also compute the IV curve::

    voltages = np.linspace(0, 0.95, 40)
    solution = sesame.solve_equilibrium(sys)
    sesame.IVcurve(sys, voltages, solution, '1dpnIV')

The data files will have names like ``1dpnIV.vapp_0.npz`` where the number 0
is the index of of the array ``voltages``. We will see how to extract the data
from these files and compute observables in :doc:`tutorial 5 <analysis>`.
