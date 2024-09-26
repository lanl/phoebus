Tracer Particles
================

``Phoebus`` includes Lagrangian tracer particles.
These are primarily for the purpose of postprocessing simulation data for, e.g., nucleosynthesis.
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
These quantites include

.. todo:: 

   Clean up below.

* Density
* Temperature
* Electron fractionb
* Internal energy
* Entropy
* 3-velocity
* Relativistic shift
* Relativistic lapse
* Lorentz factor
* Spartial metric determinant
* Pressure
* Bernoulli parameter
* Primitive magnetic field components
