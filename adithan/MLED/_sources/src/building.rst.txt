Building
========

Obtaining the Source Code
-------------------------

``phoebus`` uses submodules extensively. To make sure you get them all,
clone it as

.. code:: bash

   git clone --recursive git@github.com:lanl/phoebus.git

or as

.. code:: bash

   git clone git@github.com:lanl/phoebus.git
   cd phoebus
   git submodule update --init --recursive


Prerequisites
-------------

To build ``phoebus``, you need to create a build directly, as in-source builds are not supported.
After cloning the repository,

.. code:: bash

   cd phoebus
   mkdir bin
   cd bin

Example builds
--------------

Below are some example build commands

MPI-parallel only
~~~~~~~~~~~~~~~~~

The following will build ``phoebus`` with MPI parallelism but no shared
memory parallelism

.. code:: bash

   cmake ..
   make -j

OpenMP-parallel
~~~~~~~~~~~~~~~

The following will build ``phoebus`` with OpenMP parallelism only

.. code:: bash

   cmake -DPHOEBUS_ENABLE_MPI=Off -DPHOEBUS_ENABLE_OPENMP=ON ..
   make -j

Cuda
~~~~

The following will build ``phoebus`` with no MPI or OpenMP parallelism.

.. code:: bash

   cmake -DPHOEBUS_ENABLE_CUDA=On -DCMAKE_CXX_COMPILER=${HOME}/phoebus/external/singularity-eos/utils/kokkos/bin/nvcc_wrapper -DKokkos_ARCH_HSW=ON -DKokkos_ARCH_VOLTA70=ON -DPHOEBUS_ENABLE_MPI=OFF ..

A few notes for this one: - Note here the ``-DCMAKE_CXX_COMPILER`` flag.
This is necessary. You *must* set the compiler to ``nvcc_wrapper``
provided by Kokkos. - Note the ``-DKokkos_ARCH_*`` flags. Those set the
host and device microarchitectures and are required. The choice here is
on an ``x86_64`` machine with a ``volta`` GPU.

Build Options
-------------

The build options explicitly provided by ``phoebus`` are:

+-----------------------+----------+-----------------------------------------------+
| Option                | Default  | Comment                                       |
+=======================+==========+===============================================+
| PHOEBUS_ENABLE_CUDA   | OFF      | Cuda                                          |
+-----------------------+----------+-----------------------------------------------+
| PHOEBUS_ENABLE_HDF5   | ON       | HDF5. Required for output and restarts.       |
+-----------------------+----------+-----------------------------------------------+
| PHOEBUS_ENABLE_MPI    | ON       | MPI. Required for distributed memory          |
|                       |          | parallelism.                                  |
+-----------------------+----------+-----------------------------------------------+
| PHOEBUS_ENABLE_OPENMP | OFF      | OpenMP. Required for shared memory            |
|                       |          | parallelism.                                  |
+-----------------------+----------+-----------------------------------------------+
| MACHINE_CFG           | None     | Machine-specific config file, optional.       |
+-----------------------+----------+-----------------------------------------------+

Some relevant settings from Parthenon and Kokkos you may need to play
with are:

+---------------------+----------+---------------------------------------------+
| Option              | Default  | Comment                                     |
+=====================+==========+=============================================+
| CMAKE_CXX_COMPILER  | None     | Must be set to ``nvcc_wrapper`` with cuda   |
|                     |          | backend                                     |
+---------------------+----------+---------------------------------------------+
| Kokkos_ARCH_XXXX    | OFF      | You must set the GPU architecture when      |
|                     |          | compiling for Cuda                          |
+---------------------+----------+---------------------------------------------+

You can see all the Parthenon build options
`here <https://github.com/lanl/parthenon/blob/develop/docs/building.md>`__
and all the Kokkos build options
`here <https://github.com/kokkos/kokkos/wiki/Compiling>`__

Cmake machine configs
---------------------

If you are proficient with ``cmake`` You can optionally write a
``cmake`` file that sets the configure parameters that you like on a
given machine. Both ``phoebus`` and ``parthenon`` can make use of it.
You can point to the file with

::

   -DMACHINE_CFG=path/to/machine/file

at config time or by setting the environment variable ``MACHINE_CFG`` to
point at it, e.g.,

.. code:: bash

   export MACHINE_CFG=path/to/machine/file

An example machine file might look like

::

   # Machine file for x86_64-volta on Darwin
   message(STATUS "Loading machine configuration for Darwin x86-volta node")
   message(STATUS "Assumes: module load module load gcc/7.4.0 cuda/10.2 openmpi/4.0.3-gcc_7.4.0 anaconda/Anaconda3.2019.10 cmake && spack load hdf5")
   message(STATUS "Also assumes you have a valid spack installation loaded.")

   set(PHOEBUS_ENABLE_CUDA ON CACHE BOOL "Cuda backend")
   set(PHOEBUS_ENABLE_MPI OFF CACHE BOOL "No MPI")
   set(Kokkos_ARCH_HSW ON CACHE BOOL "Haswell target")
   set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "volta target")
   set(CMAKE_CXX_COMPILER /home/jonahm/phoebus/external/parthenon/external/Kokkos/bin/nvcc_wrapper CACHE STRING "nvcc wrapper")

you could then configure and compile as

.. code:: bash

   cmake -DMACHINE_CFG=path/to/machine/file ..
   make -j

Running
-------

Run phoebus from the ``build`` directory as

.. code:: bash

   ./src/phoebus -i path/to/input/file.pin

The input files are in ``phoebus/inputs/*``. There’s typically one input
file per problem setup file.

Submodules
----------

-  ``parthenon`` asynchronous tasking and block-AMR infrastructure
-  ``singularity-eos`` provides performance-portable equations of state
   and PTE solvers
-  ``singularity-opac`` provides performance-portable opacities and
   emissivities
-  ``Kokkos`` provides performance portable shared-memory parallelism.
   It allows our loops to be CUDA, OpenMP, or something else. By default
   we use the ``Kokkos`` shipped with ``parthenon``.

External (Required)
-------------------

-  ``cmake`` for building

Optional
--------

-  ``hdf5`` for output (must be parallel if MPI is enabled)
-  ``MPI`` for distributed memory parallelism
-  ``python3`` for visualization
