.. _singularity-eos: https://lanl.github.io/singularity-eos

CCSNe Initialization
==================

.. note::

   These docs (and this pipeline!) are under active development. If you encounter a bug or add a new feature, let the developers know!

If you're interested in exploring the core-collapse supernova (e.g. ``progenitor`` ) problem in ``Phoebus``, 
you'll need to take a stellar progenitor from a Lagrangian code (e.g. MESA, KEPLER, GR1D) and convert it to
a ``Phoebus`` -readable input file.

This Python pipeline offers a variety of options to make that conversion as simple as possible. 


How do I use this pipeline?
----------------------------

Like ``Phoebus``, this pipeline works through a master input file. You can run the entire process using the following command:

.. code-block:: bash

    python3 stellar.py --file {input file name} -v

Within the input file, there are a variety of flags and options that can be used to process your stellar progenitor.

.. list-table:: 
   :header-rows: 1
   :widths: 35 20 45

   * - Parameter
     - Default
     - Description
   * - ``‑‑model‑path``
     - ``../lawhite/mesa/15m_at_cc.dat``
     - Absolute or relative path to input model/data.
   * - ``‑‑model‑name``
     - ``aag21``
     - Desired name/id for your model.
   * - ``‑‑model‑type``
     - ``MESA``
     - Type of model (options: MESA, GR1D, KEPLER).
   * - ``‑‑model‑header``
     - ``-1``
     - (optional) Line number (zero-indexed) of the profile header (if < 0, goes to defaults - MESA: 4, KEPLER: 1).
   * - ``‑‑atbounce``
     -
     - (bool) If using a GR1D model, pulls data files at bounce.
   * - ``‑‑timestamp``
     - ``0.3``
     - (optional) If using a GR1D model and atbounce = false, gets data from timestep (in seconds).
   * - ``‑‑wma‑vel``
     -
     - (optional, bool) Enables weighted mean averaging for velocity.
   * - ``‑‑wma‑all``
     -
     - (optional, bool) Enables weighted mean averaging for all primitives in profile.
   * - ``‑‑wma‑n``
     - ``100``
     - (optional) Window for weighted mean average.
   * - ``‑‑wma‑exp``
     - ``3``
     - (optional) Exponent to raise wma weights to.
   * - ``‑‑factor‑interp``
     - ``10``
     - (optional) Factor to increase profile radial grid resolution by.
   * - ``‑‑use‑drad‑interp``
     -
     - (optional, bool) Enables radial spacing critera for interpolation instead of factor, recommended for lagrangian (mass coordinate) input models.
   * - ``‑‑drad‑interp``
     - ``1e7``
     - (optional) If using radial spacing for interpolation, sets minimum spacing requirement in cm.
   * - ``‑‑use‑uniform``
     -
     - (optional, bool) Determines if the higher resolution profile uses a uniform grid.
   * - ``‑‑unif‑max``
     - ``5.0e9``
     - (optional) If above disabled, sets outer boundary of inner uniform grid (uniform > log).
   * - ``‑‑adm‑problem``
     - ``StellarTable``
     - (optional) Type of ADM interpolation to use, default: StellarCollapse (options: StellarCollapse, tov, homologous).
   * - ``‑‑eos‑path``
     - ``../grsolver/SFHo.h5``
     - Absolute or relative path to eos table (must be \*.h5 file).
   * - ``‑‑eos‑type``
     - ``StellarCollapse``
     - Type of tabulated eos, default: StellarCollapse (options: StellarCollapse, Helmholtz).
   * - ``‑‑use‑rhocut``
     -
     - (bool) Enables a density-only cut for the new grid. Default option if no ADM grid flags are included.
   * - ``‑‑rhocut``
     - ``2.0e3``
     - (optional) If using rhocut, the density to truncate at in g/cm^3.
   * - ``‑‑use‑radcut``
     -
     - (bool) Enables a radius-only cut for the new grid; keeps original inner radius.
   * - ``‑‑radcut``
     - ``1.0e9``
     - (optional) If using radcut, the radius to truncate at in cm.
   * - ``‑‑use‑custom``
     -
     - (bool) Lets user input custom grid (radial) in cm.
   * - ``‑‑custom‑min``
     - ``5.0e5``
     - (optional) Inner radial limit for custom grid in cm.
   * - ``‑‑custom‑max``
     - ``5.0e9``
     - (optional) Outer radial limit for custom grid in cm.
   * - ``‑‑use‑def‑rad``
     -
     - (optional, bool) Uses the default input radial grid of the progenitor (best used for GR1D models at or after bounce).
   * - ``‑‑zones``
     - ``2048``
     - (optional) Desired number of grid zones for new profile. Recommended to be 2^x zones for phoebus AMR.
   * - ``‑‑interp‑method``
     - ``cubic``
     - (optional) Method of interpolation for ADM solver, non-ADM primitives (options: linear, cubic, akima, makima).
   * - ``‑‑bc‑type``
     - ``not‑a‑knot``
     - (optional) Boundary conditions if using cubic interpolation (options: clamped, not-a-knot, periodic).
   * - ``‑‑interp‑method‑adm``
     - ``piecewise``
     - (optional) Method of interpolation for ADM solver, ADM quantities (options: linear, piecewise).
   * - ``‑‑iterations``
     - ``100``
     - (optional) Number of iterations the ADM solver will try when converging.
   * - ``‑‑dalpha‑eps``
     - ``1.0e-12``
     - (optional) Allowable variation of the lapse, s.t. dalpha must converge to be less than dalpha-eps
   * - ``‑‑extrapolate``
     -
     - (optional, bool) Extrapolate the output grid to r = 0.
   * - ``‑‑do‑fixup``
     -
     - (optional, bool) Enables a FLASH-style fixup to radius (face ‑‑> cell centered) and first (inner) zone velocity.
   * - ``‑‑save‑path``
     - ``test-profiles``
     - Absolute or relative path to desired directory to save all output.
   * - ``‑‑save‑info``
     -
     - (optional, bool) Saves info file for cgs -> phb unit conversions and input.
   * - ``‑‑save‑unconverted``
     -
     - (optional, bool) Saves output for phoebus before conversion to phb units.
   * - ``‑‑save‑input``
     -
     - (optional, bool) Saves input parameter file for future use.
   * - ``‑‑save‑raw``
     -
     - (optional, bool) Saves the processed, pre ADM solver progenitor.
   * - ``‑‑depr``
     -
     - (optional, bool) FOR TESTING ONLY, uses original grsolver pipeline.

The full parameter documentation can also be found in ``phoebus/scripts/python/stellar/params.in``, which can be copied and used as a starting template.


Dependencies
````````````

For this pipeline to run, you'll need the following libraries:

- ``numpy``
- ``astropy``
- ``scipy``
- ``pandas``

You'll also need an installation of `singularity-eos`_ that's been compiled with the Python bindings enabled. Something similar to 
the following (adjusted for your HPC environment) should suffice.

.. code-block:: bash

    git clone --recursive git@github.com:lanl/singularity-eos.git
    cd singularity-eos

    # load python and hdf5/phdf5 here!

    cmake -B builddir -S . \
    -DSINGULARITY_FORCE_SUBMODULE_MODE=ON \
    -DSINGULARITY_USE_FORTRAN=OFF \
    -DSINGULARITY_USE_STELLAR_COLLAPSE=ON \
    -DSINGULARITY_BUILD_PYTHON=ON \
    -DSINGULARITY_USE_SPINER=ON

    cmake --build builddir --parallel
    mkdir install # << or install directory of your choice
    cmake --install builddir --prefix install


Documentation
-------------

This entire pipeline can be used as a "black-box" process. However, there are many functions that may be helpful for detailed analysis, 
processing, progenitor I/O, troubleshooting, and more. Detailed documentation for all classes, methods, and functions can be found below.


Progenitor Processing & I/O
````````````````````````````

.. automodule:: progenitors
    :members:
    :no-private-members:
    :member-order: bysource


Solving for ADM initial conditions
``````````````````````````````````
.. autoclass:: adm.ADMSolver
    :members:
    :special-members: __init__
    :no-private-members:
    :member-order: bysource


Converting to Phoebus-style input
`````````````````````````````````

.. automodule:: convert
    :members:
    :no-private-members:
    :member-order: bysource

There are also a few helper modules that use `singularity-eos`_ that can be used to calculate specific internal energy and get bounds for the 
equation of state in use:

.. automodule:: seos
    :members:
    :no-private-members:
    :member-order: bysource
