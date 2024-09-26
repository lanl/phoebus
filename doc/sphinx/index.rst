.. Phoebus documentation master file, created by
   sphinx-quickstart on Thu Sep 26 15:18:20 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Phoebus: Performance portable GRRMHD for supernovae, mergers, and more
======================================================================

`Phoebus`_ is a performance portable general relativistic neutrino radiation magnetohydrodynamics code built upon the `Parthenon`_ adaptive mesh refinement framework.

.. _Parthenon: https://github.com/parthenon-hpc-lab/parthenon
.. _Phoebus: https://github.com/lanl/phoebus

Key Features
^^^^^^^^^^^^^

* Finite volume GRMHD
* Neutrino transport with Monte Carlo and moment methods
* Analytic and tabulated spacetimes
* Monopolar general relativistic gravity for core-collapse supernovae
* Lagrangian tracer particles
* Support for arbitrary equations of state

.. note::

   These docs are under active development.

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :glob:

   src/*
