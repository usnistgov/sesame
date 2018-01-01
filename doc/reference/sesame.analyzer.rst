:mod:`sesame.analyzer` -- Computing densities, recombination and currents
=========================================================================

Sesame provides several functions to compute densities, recombination and other
integrated quantities to simplify the analysis of simulation data. We give a
summary of these functions below:

.. module:: sesame.analyzer
.. module:: sesame.analyzer.Analyzer
   :noindex:
.. autosummary::

   line
   band_diagram
   electron_density
   hole_density
   bulk_srh_rr
   auger_rr
   radiative_rr
   defect_rr
   total_rr
   electron_current
   hole_current
   electron_current_map
   map3D
   integrated_bulk_srh_recombination
   integrated_auger_recombination
   integrated_radiative_recombination
   integrated_defect_recombination
   full_current

All the functions are gathered in the :func:`sesame.analyzer.Analyzer` class.

.. autoclass:: sesame.analyzer.Analyzer
   :members:
