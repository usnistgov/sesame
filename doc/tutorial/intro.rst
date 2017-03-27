Physical model
----------------

Here we present the geometry with the system of coordinates that Sesame assumes,
and the set of equations that it solves.

Geometry and governing equations
.................................

Our model system is shown below. It is a semiconductor device connected to two
contacts in :math:`x=0` and :math:`x=L` (where :math:`L` is the the length of
the device in the :math:`x`-direction). The doped regions are drawn for the
example only, any doping profile can be considered.

.. image:: model.*
   :align: center


The steady state of this system under nonequilibrium conditions is treated with
the drift-diffusion-Poisson equations

.. math:: 
   \vec{\nabla}\cdot \vec{J}_n &= -q(G-R)\\
   \vec{\nabla}\cdot \vec{J}_p &= q(G-R)\\
   \Delta \phi &= \frac{\rho}{\epsilon\epsilon_0}
   :label: ddp

with the currents

.. math:: 
   \vec{J}_n &= -q\mu_n n \vec{\nabla} \phi + qD_n \vec{\nabla}n \\
   \vec{J}_p &= -q\mu_p p \vec{\nabla} \phi - qD_p \vec{\nabla}p
   :label: currents

where :math:`n, p` are the electron and hole number densities, and :math:`\phi`
is the electrostatic potential. :math:`J_{n(p)}` is the charge current density
of electrons (holes). Here :math:`q` is the absolute value of the electron
charge. :math:`\rho` is the local charge and :math:`\epsilon` is the dielectric
constant of the material. :math:`\mu_{n,p}` is the electron/hole
mobility, and is related the diffusion :math:`D _{n,p}` by :math:`D_{n,p} =
k_BT\mu_{n,p}/q`.  :math:`G` is the generation rate density, :math:`R` is the
recombination and we denote the net generation rate :math:`U=G-R`. The natural
length scale is the Debye length, given by :math:`\lambda = \epsilon_0 k_B T /(q^2
N )`, where :math:`N` is the concentration relevant to the problem. Combining
Eqs. :eq:`ddp` and Eqs. :eq:`currents`, and scaling by the Debye length leads to
the following system

.. math:: 
   \widetilde{\vec{\nabla}} \cdot \left(-\bar n \widetilde{\vec{\nabla}} \bar \phi + \widetilde{\vec{\nabla}}\bar n \right) &= \bar U

   \widetilde{\vec{\nabla}} \cdot \left(-\bar p \widetilde{\vec{\nabla}}\bar \phi - \widetilde{\vec{\nabla}}\bar p \right) &= -\bar U

   \widetilde{\Delta} \bar \phi &= \frac{\bar n - \bar p}{\epsilon}

where :math:`\widetilde{\vec{\nabla}}` is the dimensionless spatial first derivative
operator, :math:`\widetilde{\Delta}` is the dimensionless Laplacian and 
the dimensionless variables are

.. math::
   \bar \phi &= \frac{e\phi}{k_BT}\\
   \bar n &= n/N \\
   \bar p &= p/N \\
   \bar U &= \frac{U \lambda^2}{ND} \\
   \bar t &= t \frac{q\mu N}{\epsilon_0} \\
   \bar J_{n,p} &= J_{n,p} \frac{qD_{n,p}N}{\lambda} 

We suppose that the recombination is through three mechanims:
Shockley-Read-Hall, radiative and Auger.  Unlike the defect-mediated
recombination, radiative and Auger processes are respectively second and third
order in charge density. For this reason they become more important at higher
densities (such as for heavily doped systems, or systems with high generation
rate density of electron-hole pairs).  On the other hand, Shockley-Read-Hall
recombination is first order in charge density.

The Shockley-Read-Hall takes the form

.. math::
   R_{\rm SRH} = \frac{np - n_i^2}{\tau_p(n+n_1) + \tau_n(p+p_1)}
   
where :math:`n^2_i = N_C N_V e^{-E_g/k_BT}, n_1 = N_C e^{-(E_C - E_T) /k_BT} ,
p_1 = N_V e^{- (E_T - E_V) /k_BT}`, where :math:`E_T` is the
energy level of the trap state. The above can be derived on very general grounds
(see Ref. [1]_, page 34). :math:`\tau_{n,(p)}` is the bulk lifetime for
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

Boundary conditions at the contacts
...................................
The boundary conditions for carriers at charge-collecting contacts located at
:math:`x=0` and `x = L` are typically
parameterized with the surface recombination velocities for electrons and holes
at the contacts, denoted respectively by :math:`S_{c_p}` and :math:`S_{c_n}`

.. math::
   \vec{J}_n(0,y,z) \cdot \vec{u}_x &= qS_{c_n} (n(0,y,z) - n_{\rm eq}(0,y,z))\\
   \vec{J}_p(0,y,z) \cdot \vec{u}_x &= -qS_{c_p} (p(0,y,z) - p_{\rm eq}(0,y,z))\\
   \vec{J}_n(L,y,z) \cdot \vec{u}_x &= -qS_{c_n} (n(L,y,z) - n_{\rm eq}(L,y,z))\\
   \vec{J}_p(L,y,z) \cdot \vec{u}_x &= -qS_{c_p} (p(L,y,z) - p_{\rm eq}(L,y,z))\\
   :label: BCs

where :math:`n(p)_{\rm eq}` is the thermal equilibrium electron (hole) density.
In being collected by a contact, a carrier in the valence or conduction band
loses its energy and relaxes to the chemical potential of the bulk contact.
This is essentially a recombination process. The recombination velocity may be
thought of identically as a carrier lifetime in twho dimension (considering
Eq. :eq:`tau` where the trap density is two-dimensional, one obtains units of
velocity).  

Additional charges: line and plane defects
............................................
Additional charged defects can be added to the system to simulate, for example,
grain boundaries in a semiconductor. Grain boundaries separate crystallites of
different orientation. They occupy a reduced dimensionality space (e.g. 2D
planes embedded in a 3D material), and typically possess a high density of
defects (dangling bonds, wrong bonds, \cdots). These lead to localized states
within the gap, which may lead to charging of the grain boundary, and increased
recombination at the grain boundary.

Considering energy levels with charge transitions :math:`\alpha_j/\beta_j`, the
charge density at the defect sites reads

.. math::
    Q = q\sum_j \rho_j (\mathrm{max}(\alpha_j, \beta_j) (1-f_j) + \mathrm{min}(\alpha_j, \beta_j) f_j)

where :math:`\rho_j` is the 2D defect density of state at energy :math:`E_j`.
The occupancy of the `j`-th defect level :math:`f_j` is given by [2]_

.. math::
    f_j = \frac{S_n n_{\rm GB} + S_p \bar p_j}{S_n(n_{\rm GB}+\bar n_j) + S_p(p_{\rm GB}+\bar p_j)} 

where :math:`n_{\rm GB}` (:math:`p_{\rm GB}`) is the electron (hole) density at the
grain boundary, :math:`S_n`, :math:`S_p` are recombination velocity parameters for electrons
and holes respectively. :math:`\bar n_j` and :math:`\bar p_j` are

.. math::
   \bar n_j &= N_C e^{\left(-E_g/2 + E_j\right)/k_BT}\\
   \bar p_j &= N_V e^{(-E_g/2 - E_j)/k_BT}

where :math:`E_j` is calculated from the valence band edge, :math:`N_C`
(:math:`N_V`) is the conduction (valence) band effective density of states.


The increased recombination at the grain boundary is included by an additional
recombination term :math:`R_{\rm GB}` at the grain boundary core

.. math::
   R_{\rm GB} = \sum_j \frac{S_nS_p(n_{\rm GB} p_{\rm GB} - n_i^2)}
   {S_n(n_{\rm GB} + \bar n_j) + S_p(p_{\rm GB} + \bar p_j)}

Embedding a two-dimensional density into the three-dimensional model is formally
accomplished with the use of a delta function. Numerically, the two-dimensional
defect densities of states and the surface recombination velocities are divided
by the size of the discretized grid :math:`dl` at the position of the plane, and along
the direction normal to the plane.


Carrier densities and quasi-Fermi levels
........................................

Despite their apparent simplicity, Eqs. :eq:`ddp`, and the set of boundary
conditions of the form of Eq. :eq:`BCs` are
numerically challenging to solve. This is due in part to the fact that the
carrier densities vary by many orders of
magnitude throughout the sample, and because drift and diffusion currents often
nearly cancel each other, and the
entire solution depends on the small residual current left over. We next discuss
a slightly different form of these
same equations which is convenient to use for numerical solutions. We introduce
the concept of quasi-Fermi level for
electrons and holes (denoted by :math:`E_{F_n}` and :math:`E_{F_p}`  respectively). The carrier
density is related to these quantities as 

.. math::
   n(x,y,z) &= N_C e^{\left(E_{F_n}(x,y,z) + q\phi(x,y,z) - b_l\right)/k_BT}\\
   p(x,y,z) &= N_V e^{\left(E_{F_p}(x,y,z) - q\phi(x,y,z) - E g +b_l\right)/k_BT}
   :label: np

where the term :math:`b_l` essentially sets (or is set by) the zero of energy
for the electrostatic potential (the default value is 0).  Quasi-fermi levels
are convenient in part because they guarantee that carrier densities are always
positive. While carrier densities vary by many orders of magnitude, quasi-Fermi
levels require much less variation to describe the system. The signs in Eq.
:eq:`np` can be confusing; they are such that the carrier density is larger if
its quasi-Fermi level is more positive, see the figure below.

.. figure:: bands.*
   :align: center
   :figwidth: 500

   Equilibrium energy level diagrams showing the electron quasi-Fermi
   level in an n-type (p-type) semiconductor on the left (right). We chose
   :math:`q\phi = -E_g/2` to make electron and hole quasi-Fermi levels
   symmetric. 

On an energy diagram, this
means that more positive electron quasi-Fermi levels are plotted closer to the
conduction band, while positive hole quasi-Fermi levels are plotted closer to
the valence band. When plotting both electron and hole quasi-Fermi levels on the
same graph (such as on a band diagram), it is therefore necessary to plot, for
example, :math:`E_{F_n}` and :math:`-E_{F_p}` in order to have a consistent sign
convention for reading the plot.  Signs are confusing additionally because
people use different conventions, and sometimes people are careless about them.
The signs utilized in these notes and in the code have been checked, and are all
self-consistent. Any questions about signs should therefore not be ascribed to
typos and the like.

The electron and hole current can be shown to be proportional to the spatial
gradient of the quasi-Fermi level

.. math::
   \vec{J}_n &= q\mu_n n \vec{\nabla} E_{F_n}\\
   \vec{J}_p &= -q\mu_p p \vec{\nabla} E_{F_p}

These relations for the currents will be used in the discretization of Eq.
:eq:`ddp`.

.. rubric:: References
.. [1] S. J. Fonash, *Solar cell device physics*, Academic Press 1981.
.. [2] W. Shockley, W. T. Read, Jr., *Phys. Rev.*, **87**, 835 (1952).
