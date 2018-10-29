.. _solvers_doc:

:mod:`sesame.solvers` -- Equilibrium and nonequilibrium solvers
===============================================================

Sesame offers several solvers to address quickly specific problems (equilibrium
potential, IV curve) without having to manage a cumbersome machinery to get to
the solution.

:mod:`sesame.solvers.default` -- Default solver
-----------------------------------------------

The functions below belong to the default solver `sesame.solvers.default` which
is created upon loading sesame. We made these functions available once sesame is
loaded making the solver completely transparent for the user. The solver stores
the equilibrium potential of the system as soon as it has been computed.

.. module:: sesame.solvers
.. module:: sesame.solvers.default

.. autosummary::
   :toctree: generated/

   solve
   IVcurve


Creating  another solver
------------------------

Making another solver is done by creating an instance of the
`sesame.solvers.Solver` class. This can be used to turn off the use of the MUMPS
library even when the library is available.

.. toctree::
   :maxdepth: 1

   sesame.solvers.solver

