Numerical treatment of the drift diffusion Poisson equations
============================================================

In this section we present the procedure followed to discretize the drift
diffusion Poisson set of equations, the algorithm used to solve it and its
implementation.

Scharfetter-Gummel scheme
-------------------------
To solve the drift diffusion Poisson equations numerically, we utilize a simple
spatial discretization.  Recall that densities are defined on sites, and fluxes
(such as current flux, electric field flux) are defined on links.  It's important to note that *sites* and
*links* in the discretized grid are fundamentally different objects, as shown in
the figure below.

.. figure:: grid_link.*
   :align: center
   :width: 300	

   Sites versus links.  We take the indexing convention that :math:`\Delta
   x^i` represents the space between sites :math:`i` and :math:`i+1`.

We consider a one-dimensional system to illustrate how the discretization of the
model is done.  First, we want to rewrite the currents in semi-discretized form
for link :math:`i` (link :math:`i` connects discretized points :math:`i` and
:math:`i+1`):  

.. math::
    J_n^i & = q\mu_n n_i \frac{\partial E_{F_n,i}}{\partial x} \\
    J_p^i & = q\mu_p p_i \frac{\partial E_{F_p,i}}{\partial x}
   :label: ji

Note that here, link indices are denoted with a superscript, while point indices
are denoted with a subscript.

Next, a key step to ensure numerical stability is to integrate the above in order to
get a completely discretized version of the current :math:`J^i`.  This discretization
is known as the Scharfetter-Gummel scheme [3]_, and is mandatory.  Let us
do the hole case.  First, rewrite the hole density in terms of the quasi-Fermi
level. 

.. math::
    p_i = N_V e^{\left(b_l-E_g+E_{F_p}(x)-\phi(x)\right)/k_BT}

We plug this form of :math:`p` into Eq. :eq:`ji`, then multiply both sides of
the hole current  by :math:`e^{q\phi(x)/k_BT}\ dx`, 

.. math::
    J_p^i = q \mu_p N_V e^{\left(b_l-E_g-E_{F_p}-q\phi(x)\right)/k_BT}
    \frac{\partial E_{F_p}}{\partial x} 
    
and integrate over link :math:`i`

.. math::
    \int J_p^i e^{q\phi(x)/k_BT} \mathrm{d}x
    = q \mu_p N_V e^{\left(b_l-E_g\right)/k_BT} \int e^{-E_{F_p}/k_BT}
    \mathrm{d}E_{F_p}
   :label: eqx

Now we assume that the potential varies linearly between grid points, 

.. math::
    \phi \left(x\right) = \frac{\phi_{i+1}-\phi_{i}}{\Delta x^i}\left(x-x_i\right)+\phi_i,

which enables the integral on the left hand side above to be performed:

.. math::
    \int_{x_i}^{x_{i+1}} \mathrm{d}x e^{\pm q\phi(x)/k_BT} = \pm
    \frac{k_BT}{q} \Delta x^i \frac{e^{\pm q\phi_{i+1}/k_BT} - e^{\pm
    q\phi_i / k_BT}}{\phi_{i+1} - \phi_i}
   :label: eqx2

Plugging Eq. :eq:`eqx2` into Eq. :eq:`eqx` and solving for :math:`J_p^i` yields

.. math::
    J_p^i = \frac{q^2/k_BT}{\Delta x^i}
    \frac{\phi_{i+1}-\phi_i}{e^{q\phi_{i+1}/k_BT}-e^{q\phi_i/k_BT}} 
    \mu_p N_V e^{\left(b_l-E_g\right)/k_BT} \left[e^{-E_{F_p,i+1}/k_BT}-e^{-E_{F_p,i}}\right]
   :label: jpi

A similar procedure leads to the following expression for :math:`J_n^i`:

.. math::
    J_n^i = \frac{q^2/k_BT}{\Delta x^i}
    \frac{\phi_{i+1}-\phi_i}{e^{-q\phi_{i+1}/k_BT}-e^{-q\phi_i/k_BT}}
    \mu_n N_C e^{-b_l}  \left[e^{E_{F_n,i+1}/k_BT}-e^{E_{F_n,i}/k_BT}\right]
   :label: jni

The formulations of :math:`J_{n,p}^i` given in Eqs. :eq:`jpi` and :eq:`jni`
ensure perfect current conservation.




.. _algo:

Newton-Raphson algorithm
------------------------
We want to write the continuity and Poisson equations in the form :math:`f(x)=0`,
and solve these coupled nonlinear equations by using root-finding algorithms.
The appropriate form is given by: 

.. math::
    f_p^i &= \frac{2}{\Delta x^i + \Delta x^{i-1}}\left(J_p^{i} -
    J_p^{i-1}\right) + G_i - R_i 
    \\ f_n^i &= \frac{2}{\Delta x^i + \Delta
    x^{i-1}}\left(J_n^{i} - J_n^{i-1}\right) - G_i + R_i \\ 
    f_v^i &= \frac{2}{\Delta x^i + \Delta x^{i-1}}
    \left( \left(\frac{\phi_{i}-\phi_{i-1}}{\Delta x^{i-1}}\right)
    -\left(\frac{\phi_{i+1}-\phi_i}{\Delta x^i}\right) \right) -
    \frac{\rho_i}{\epsilon}

These equations are the
discretized drift-diffusion-Poisson equations to be solved for the variables
:math:`\left\{E_{F_n,i}, E_{F_p,i}, \phi_i\right\}`, subject to the boundary
conditions given in introduction.


We use a Newton-Raphson method to solve the above set of equations.  The idea
behind the method is clearest in a simple one-dimensional case as illustrated on
the figure below.  Given a general nonlinear function :math:`f(x)`, we want to find its
root :math:`\bar x: f(\bar x)=0`.  Given an initial guess :math:`x_1`, one can
estimate the error :math:`\delta x` in this guess, assuming that the function
varies linearly all the way to its root

.. math::
    \delta x= \left(\frac{df}{dx} (x_1)\right)^{-1}f\left(x_1\right)
    :label: eq1d

An updated guess is provided by :math:`x_2 = x_1 - \delta x`.

.. figure:: NR.*
    :align: center

    Schematic for the Newton-Raphson method for root finding.

In multiple dimensions the last term in Eq. :eq:`eq1d` is replaced by the
inverse of the Jacobian, which is the multi-dimensional generalization
of the derivative.  In this case, Eq. :eq:`eq1d` is a matrix equation of
the form: 

.. math::
    \delta {\bf x} = A^{-1} {\bf F}\left({\bf x}\right)

where

.. math::
    A_{ij} = \frac{\partial F_i}{\partial x_j}

Here is a small subset of what the :math:`A` matrix looks like for our problem.
We have only explicitly shown the row which corresponds to :math:`f_n^i` (here we
drop the super/sub script convention set up to distinguish between
sites and links, for the sake of writing things more compactly):

.. math::
    \left(
    \begin{array}{ccccccccccc}
      & \ldots &  &  &  &  &  &  & & &\\
      \vdots  &  &  &  &  &  &  &  & & &  \\
       &  &  &  &  &  &  &  &  & &\\
       &  &  &  &  &  &  &  &  & &\\
      \ldots & \frac{\partial f_n^i}{\partial E_{F_n}^{i-1}} & \frac{\partial
      f_n^i}{\partial E_{F_p}^{i-1}}  & \frac{\partial f_n^i}{\partial \phi^{i-1}}
      & \frac{\partial f_n^i}{\partial E_{F_n}^{i}} & \frac{\partial
      f_n^i}{\partial E_{F_p}^{i}}  & \frac{\partial f_n^i}{\partial \phi^{i}}  &
      \frac{\partial f_n^i}{\partial E_{F_n}^{i+1}} & \frac{\partial
      f_n^i}{\partial E_{F_p}^{i+1}}  & \frac{\partial f_n^i}{\partial \phi^{i+1}} &
      \ldots \\ \vdots &  &  &  &  &  &  &  & & &\\
       &  &  &  &  &  &  &  &  & &\\
       &  &  &  &  &  &  &  &  & &\\
       &  &  &  &  &  &  &  &  & &\\
       &  &  &  &  &  &  &  &  & &\\
       & \ldots &  &  &  &  &  &  &  & &
    \end{array}
    \right)
    \left(
      \begin{array}{c}
      \vdots\\
        \delta E_{F_n}^{i-1} \\
        \delta E_{F_p}^{i-1} \\
        \delta \phi^{i-1} \\
        \delta E_{F_n}^{i} \\
        \delta E_{F_p}^{i} \\
        \delta \phi^{i} \\
        \delta E_{F_n}^{i+1} \\
        \delta E_{F_p}^{i+1} \\
        \delta \phi^{i+1} \\
        \vdots
      \end{array}
    \right)
    =
    \left(
      \begin{array}{c}
      \vdots\\
        f_n^{i-1} \\
        f_p^{i-1} \\
        f_v^{i-1} \\
        f_n^{i} \\
        f_p^{i} \\
        f_v^{i} \\
        f_n^{i+1} \\
        f_p^{i+1} \\
        f_v^{i+1} \\
        \vdots
      \end{array}
    \right)
    :label: corr

Note that for this
problem, finding derivatives numerically leads to major convergence problems. We
derived the derivatives and implemented them in the code for this reason.





Multi-dimensional implementation
--------------------------------
We do the standard *folding* of the multi-dimensional index label :math:`(i,j,k)`
into the single index label :math:`s` of the sites of the system: 

.. math::
    s = i + (j \times n_x) + (k \times n_x n_y)

where :math:`n_x` (:math:`n_y`) is the number of sites in the
:math:`x`-direction (:math:`y`-direction).

Using sparse matrix techniques is key fast to fast computation. We provide below
the number of non-zero elements in the Jacobian for periodic boundary conditions
in the :math:`y`- and :math:`z`-directions.

+------------------------+-------------------------------------------------------+
| Dimension              | Number of stored values in the Jacobian               |
+========================+=======================================================+
|          1             |  19 (n\ :sub:`x`-2) + 20                              |
+------------------------+-------------------------------------------------------+
|          2             |  n\ :sub:`y` [29 (n\ :sub:`x` - 2) + 28]              |
+------------------------+-------------------------------------------------------+
|          3             |  n\ :sub:`y` n\ :sub:`z` [39 (n\ :sub:`x` - 2) + 36]  |
+------------------------+-------------------------------------------------------+

By default the Newton correction is computed by a direct resolution of the
system in Eq. :eq:`corr`. This is done using the default Scipy solver which gives
quite poor performances. We recommend using the MUMPS library instead. Note that
for large systems, and especially for 3D problems, the memory and the computing
time required by the direct methods aforementioned become so large that they are
impractical. It is possible to use an iterative method to solve Eq. :eq:`corr` in
these cases.




.. rubric:: References
.. [3] H. K. Gummel, IEEE Transactions on Electron Devices, **11**, 455 (1964).
