Physical model
----------------

Here we present the geometry with the system of coordinates that Sesame assumes,
and the set of equations that it solves.

Geometry and governing equations
.................................

Our model system is shown below. It is a semiconductor device connected to two
contacts at :math:`x=0` and :math:`x=L`. The doped regions are drawn for the
example only, any doping profile can be considered.

.. image:: model.*
   :align: center


The steady state of this system under nonequilibrium conditions is described by
the drift-diffusion-Poisson equations:

.. math:: 
   \vec{\nabla}\cdot \vec{J}_n &= -q(G-R)\\
   \vec{\nabla}\cdot \vec{J}_p &= q(G-R)\\
   \vec{\nabla}\cdot (\epsilon\vec{\nabla} \phi) &= -\rho/\epsilon_0
   :label: ddp

with the currents

.. math:: 
   \vec{J}_n &= -q\mu_n n \vec{\nabla} \phi + qD_n \vec{\nabla}n \\
   \vec{J}_p &= -q\mu_p p \vec{\nabla} \phi - qD_p \vec{\nabla}p
   :label: currents

where :math:`n, p` are the electron and hole number densities, and :math:`\phi`
is the electrostatic potential. :math:`J_{n(p)}` is the charge current density
of electrons (holes). Here :math:`q` is the absolute value of the electron
charge. :math:`\rho` is the local charge, :math:`\epsilon` is the dielectric
constant of the material, and :math:`\epsilon_0` is the permittivity of free space. :math:`\mu_{n,p}` is the electron/hole
mobility, and is related the diffusion :math:`D _{n,p}` by :math:`D_{n,p} =
k_BT\mu_{n,p}/q`, where :math:`k_B` is Boltzmann's constant and :math:`T` is the temperature.  :math:`G` is the generation rate density, :math:`R` is the
recombination and we denote the net generation rate :math:`U=G-R`. The natural
length scale is the Debye length, given by :math:`\lambda = \epsilon_0 k_B T /(q^2
N )`, where :math:`N` is the concentration relevant to the problem. Combining
Eqs. :eq:`ddp` and Eqs. :eq:`currents`, and scaling by the Debye length leads to
the following system

.. math:: 
   \widetilde{\vec{\nabla}} \cdot \left(-\bar n \widetilde{\vec{\nabla}} \bar \phi + \widetilde{\vec{\nabla}}\bar n \right) &= \bar U

   \widetilde{\vec{\nabla}} \cdot \left(-\bar p \widetilde{\vec{\nabla}}\bar \phi - \widetilde{\vec{\nabla}}\bar p \right) &= -\bar U

   \widetilde{\vec{\nabla}} \cdot (\epsilon \vec{\nabla} \bar \phi) &= (\bar n - \bar p) + (\bar{N_A} - \bar{N_D})

where :math:`\widetilde{\vec{\nabla}}` is the dimensionless spatial first
derivative operator.  :math:`\bar{N}_{A,(D)}` are the dimensionless ionized acceptor (donor) impurity concentration.  The dimensionless variables are given below:

.. math::
   \bar \phi &= \frac{q\phi}{k_BT}\\
   \bar n &= n/N \\
   \bar p &= p/N \\
   \bar U &= \frac{U \lambda^2}{ND} \\
   \bar t &= t \frac{q\mu N}{\epsilon_0} \\
   \bar J_{n,p} &= J_{n,p} \frac{\lambda}{qDN} 

with :math:`D=k_BT\mu/q` a diffusion coefficient corresponding to our choice of
scaling for the mobility :math:`\mu=1~\mathrm{cm^2/(V\cdot s)}`. See the 
:func:`~sesame.builder.Scaling` class for the implementation of these scalings.


We suppose that the bulk recombination is through three mechanisms:
Shockley-Read-Hall, radiative and Auger.  The Shockley-Read-Hall recombination takes the form

.. math::
   R_{\rm SRH} = \frac{np - n_i^2}{\tau_p(n+n_1) + \tau_n(p+p_1)}
   
where :math:`n^2_i = N_C N_V e^{-E_g/k_BT}, n_1 = n_i e^{E_T /k_BT} ,
p_1 = n_i e^{- E_T /k_BT}`, where :math:`E_T` is the
energy level of the trap state measured from the intrinsic energy level, :math:`N_C` (:math:`N_V`) is the conduction (valence) band effective density of
states. The equilibrium Fermi energy at which
:math:`n=p=n_i` is the intrinsic energy level :math:`E_i`.
:math:`\tau_{n,(p)}` is the bulk lifetime for
electrons (holes). It is given by

.. math::
   \tau_{n,p} = \frac{1}{N_T v^{\rm th}_{n,p} \sigma_{n,p}}
   :label: tau

where :math:`N_T` is the three-dimensional trap density, :math:`v^{\rm
th}_{n,p}` is the thermal velocity of carriers: :math:`v^{\rm th}_{n,p} = 3k_BT
/m_{n,p}`, and :math:`\sigma_{n,p}` is the capture cross-section for (electrons,
holes).   

The radiative recombination has the form

.. math::
   R_{\rm rad} = B (np - n_i^2)

where :math:`B` is the radiative recombination coefficient of the material. The
Auger mechanism has the form

.. math::
   R_{\rm A} = (C_n n + C_p p) (np - n_i^2)

where :math:`C_n` (:math:`C_p`) is the electron (hole) Auger coefficient.

Extended line and plane defects
...............................

Additional charged defects can be added to the system to simulate, for example,
grain boundaries or sample surfaces in a semiconductor. These extended planar
defects occupy a reduced dimensionality space: a point in a 1D model, a line in
a 2D model). The extended defect energy level spectrum
can be discrete or continuous. For a discrete spectrum, we label a defect with
the subscript :math:`d`. The occupancy of the defect level :math:`f_d` is given
by [1]_

.. math::
    f_d = \frac{S_n n + S_p p_d}{S_n(n+n_d) + S_p(p+p_d)} 

where :math:`n` (:math:`p`) is the electron (hole) density at the
defect location, :math:`S_n`, :math:`S_p` are recombination velocity parameters
for electrons and holes respectively. :math:`n_d` and :math:`p_d` are

.. math::
   \bar n_d &= n_i e^{E_d/k_BT}\\
   \bar p_d &= n_i e^{-E_d/k_BT}

where :math:`E_d` is calculated from the intrinsic Fermi level :math:`E_i`.
The defect recombination is of Shockley-Read-Hall form:

.. math::
   R_d = \frac{S_nS_p(n p - n_i^2)}{S_n(n + n_d) + S_p(p + p_d)}.

The charge density :math:`q_d` given by a single defect depends on the defect type (acceptor
or donor)

.. math::
   q_d = q\rho_d \times \left\{
    \begin{array}{ll}
        (1-f_d) & \mbox{donor} \\
        (-f_d) & \mbox{acceptor}
    \end{array}
    \right.

where :math:`\rho_d` is the defect density of state at energy :math:`E_d`.
:math:`S_n, S_p` and :math:`\rho_d` are related to the electron and hole capture
cross sections :math:`\sigma_n, \sigma_p` of the defect level by :math:`S_{n,p}
= \sigma_{n,p}v^{\rm th}_{n,p}\rho_d`, where :math:`v^{\rm th}_{n,p}` is the
electron (hole) thermal velocity.
Multiple defects are described by summing over defect label :math:`d`, or
performing an integral over a continuous defect spectrum.



Carrier densities and quasi-Fermi levels
........................................
Despite their apparent simplicity, Eqs. :eq:`ddp` are numerically challenging to
solve. We next discuss a slightly different form of
these same equations which is convenient to use for numerical solutions. We
introduce the concept of quasi-Fermi level for electrons and holes (denoted by
:math:`E_{F_n}` and :math:`E_{F_p}`  respectively). The carrier density is
related to these quantities as 

.. math::
   n(x,y,z) &= N_C e^{\left(E_{F_n}(x,y,z) + q\phi(x,y,z) + \chi(x,y,z)\right)/k_BT}\\
   p(x,y,z) &= N_V e^{\left(-E_{F_p}(x,y,z) - q\phi(x,y,z) - E_g-\chi(x,y,z)\right)/k_BT}
   :label: np

where the term :math:`\chi` is the electron affinity, :math:`\phi` is the
electrostatic potential, and :math:`E_g` is the bandgap. Note that all of these quantities may vary with position. Quasi-Fermi levels are convenient in part because they
guarantee that carrier densities are always positive. While carrier densities
vary by many orders of magnitude, quasi-Fermi levels require much less variation
to describe the system. 

The electron and hole current can be shown to be proportional to the spatial
gradient of the quasi-Fermi level

.. math::
   \vec{J}_n &= q\mu_n n \vec{\nabla} E_{F_n}\\
   \vec{J}_p &= q\mu_p p \vec{\nabla} E_{F_p}

These relations for the currents will be used in the discretization of Eq.
:eq:`ddp`.

Boundary conditions at the contacts
...................................

Equilibrium boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For a given system, Sesame first solves the equilibrium problem. In equilibrium,
the quasi-Fermi level of electrons and holes are equal and spatially
constant.  We choose an energy reference such that in equilibrium,
:math:`E_F=E_{F_p} = E_{F_n} = 0`. The equilibrium problem is therefore
reduced to a single variable :math:`\phi`. Sesame employs both
Dirichlet and Neumann equilibrium boundary conditions
for :math:`\phi`, which we discuss next.


Dirichlet boundary conditions 
"""""""""""""""""""""""""""""
Sesame uses Dirichlet boundary conditions as the
default. This is the appropriate choice when the equilibrium charge
density at the contacts is known *a priori*, and applies for Ohmic and ideal
Schottky contacts. For Ohmic boundary conditions, the carrier density is assumed
to be equal and opposite to the ionized dopant density at the contact. For an
n-type contact with :math:`N_D` ionized donors at the :math:`x = 0` contact, Eq.
:eq:`np` yields the expression for :math:`\phi^{eq}(x = 0)`:

.. math::
    \phi^{eq} (0,y,z) = k_BT \ln\left(N_D/N_C \right) -  \chi(0,y,z)

Similar reasoning yields expressions for :math:`\phi^{eq}` for p-type doping and
at the :math:`x = L` contact.  For Schottky contacts, we assume that the Fermi
level at the contact is equal to the Fermi level of the metal.  This implies
that the equilibrium electron density is :math:`N_C \exp [-(\Phi_M-\chi)/k_BT]`
where :math:`\Phi_M` is the work function of the metal contact. Eq. :eq:`np`
then yields the expression for :math:`\phi^{eq}` (shown here for
the :math:`x = 0` contact):

.. math::
    \phi^{eq} (0,y,z) = -\Phi_M|_{x=0~contact}

An identical expression applies for the :math:`x = L` contact.

Neumann boundary conditions
"""""""""""""""""""""""""""
Sesame also has an option for Neumann boundary conditions, where it is assumed
that the electrostatic field at the contact vanishes:

.. math::
   \frac{\partial \phi^{eq}}{\partial x}(0, y, z) = \frac{\partial \phi^{eq}}{\partial x}(L, y, z) = 0
   :label: bc1

The equilibrium potential :math:`\phi^{eq}` determines the equilibrium
densities :math:`n_{eq}, p_{eq}` according to Eqs. :eq:`np` with :math:`E_{F_n}
= E_{F_p} = 0`.


Out of equilibrium boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Out of thermal equilibrium, we impose Dirichlet boundary conditions on the
electrostatic potential. For example, in the presence of an applied bias
:math:`V` at :math:`x=L`, the boundary conditions are

.. math::
   \phi(0, y, z) &= \phi^{eq}(0,y,z)\\
   \phi(L, y, z) &= \phi^{eq}(L,y,z) + qV


For the drift-diffusion equations, the boundary conditions for carriers at
charge-collecting contacts are typically parameterized with the
surface recombination velocities for electrons and holes at the contacts,
denoted respectively by :math:`S_{c_p}` and :math:`S_{c_n}`

.. math::
   \vec{J}_n(0,y,z) \cdot \vec{u}_x &= qS_{c_n} (n(0,y,z) - n_{\rm eq}(0,y,z))\\
   \vec{J}_p(0,y,z) \cdot \vec{u}_x &= -qS_{c_p} (p(0,y,z) - p_{\rm eq}(0,y,z))\\
   \vec{J}_n(L,y,z) \cdot \vec{u}_x &= -qS_{c_n} (n(L,y,z) - n_{\rm eq}(L,y,z))\\
   \vec{J}_p(L,y,z) \cdot \vec{u}_x &= qS_{c_p} (p(L,y,z) - p_{\rm eq}(L,y,z))\\
   :label: BCs

where :math:`n(p)_{\rm eq}` is the thermal equilibrium electron (hole) density.



.. rubric:: References
.. [1] W. Shockley, W. T. Read, Jr., *Phys. Rev.*, **87**, 835 (1952).
