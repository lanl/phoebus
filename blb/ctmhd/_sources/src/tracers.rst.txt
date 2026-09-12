.. _parthenon-swarms: https://parthenon-hpc-lab.github.io/parthenon/develop/src/particles.html

Tracer Particles
================

``Phoebus`` includes Lagrangian tracer particles.
These are primarily for the purpose of post-processing simulation data for, e.g., nucleosynthesis.
They are operator spit from the hydrodynamics and advected with a standard second order Runge-Kutta
integrator. As ``Phoebus`` is a general relativistic code, we evolve tracer positions using a
relativistic advection equation:

.. math::

   \frac{dx^i}{dt} = \frac{u^i}{u^0} = \alpha v^i - \beta^i

Tracers may be enabled by in the ``Phoebus`` input deck as follow:

.. code-block::

   <physics>
   tracers = true

   <tracers>
   num_tracers = 1024
   defrag_frac = 0.2

Where `<physics>/tracers = true` enables tracer particles for applicable problems, 
`<tracers>/num_tracers` sets the number of tracer particles (usually per block), if applicable, 
and `<tracers>/defrag_frac` sets the fractional occupancy of tracer swarm containers.
.. note::
   Particles typically loop from 0 to some `max_active_index`.
   If only a small fraction of particles in that range are active, this 
   is inefficient. `Defrag` copies active particles to a contiguous range.
   It can be inefficient, so should be done sparingly.
Similarly, tracers may be output by modifying an existing Parthenon output block, or creating a new one:

.. code-block::

   <parthenon/output1>
   // ...
   swarms = tracers
   tracers_variables = rho, temperature, ye

   file_type = hdf5
   dt = // output cadence

Note that the position variables x, y, z are output for all swarms by default.
See the `Parthenon docs <parthenon-swarms>`_ for more information.

Tracers must, however, be configured in the problem generator.
In the generator, the tracers should be distributed through the domain, assigned positions
and unique ids. An example from the advection pgen is shown below.

.. code-block:: c++

  pmb->par_for(
      "ProblemGenerator::Advection::DistributeTracers", 0, max_active_index,
      KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          auto rng_gen = rng_pool.get_state();

          // sample in ball
          Real r2 = 1.0 + rin * rin; // init > rin^2
          while (r2 > rin * rin) {
            x(n) = x_min + rng_gen.drand() * (x_max - x_min);
            y(n) = y_min + rng_gen.drand() * (y_max - y_min);
            z(n) = z_min + rng_gen.drand() * (z_max - z_min);
            r2 = x(n) * x(n) + y(n) * y(n) + z(n) * z(n);
          }
          id(n) = num_tracers_total * gid + n;

          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
          rng_pool.free_state(rng_gen);
        }
      });

In addition to position, tracers track a number of potentially useful quantities.
These quantities include (bold quantities are 3-vectors with components _x, _y, _z)

=================== ================ ===================
   Quantity           SwarmVar Name      Description
=================== ================ ===================
Density               rho             Primitive density
Temperature           temperature     Temperature
Ye                    ye              Electron fraction (0 if ye is disabled)
Internal energy       energy          Primitive internal energy
Entropy               entropy         Entropy
**Velocity**           vel_x, ...      Three velocity
Lorentz factor        lorentz         Relativistic Lorentz factor
Lapse                 lapse           Relativistic lapse
Metric determinant    detgamma        Spatial metric determinant
**Shift**             shift_x, ...    Relativistic shift
Mass                  mass            Tracer mass
Total energy          bernoulli       Total energy Bernoulli quantity
**Magnetic field**    B_x, ...        Primitive 3-magnetic field components (if mhd enabled)
=================== ================ ===================

To minimize unnecessary work, these quantities are only populated before output using ``phoebus::UserWorkBeforeOutput``.

.. cpp:function:: void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin)
   :no-contents-entry:

   Describes work to be done prior to HDF5 output, such as populating tracer variables.
   This is connected to the appropriate Parthenon function.

