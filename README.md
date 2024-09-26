phoebus
===
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/lanl/phoebus/tests.yml?branch=main)
[![Documentation](https://github.com/lanl/phoebus/actions/workflows/docs.yml/badge.svg?branch=main)](https://lanl.github.io/phoebus/main/index.html)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Phifty One Ergs Blows Up a Star

# Obtaining the Source Code

`phoebus` uses submodules extensively. To make sure you get them all, clone it as
```bash
git clone --recursive git@github.com:lanl/phoebus.git
```
or as
```bash
git clone git@github.com:lanl/phoebus.git
cd phoebus
git submodule update --init --recursive
```

# Building

## Prerequisites

Here are a few tips and tricks for using and building phoebus. The first
step is creating a build directory, as in-source builds are
forbidden. Assuming you just cloned phoebus,
```bash
cd phoebus
mkdir bin
cd bin
```

You also need to make sure you have appropriate dependencies in your
path. On Darwin, loading the following modules (in this order) will
work:
```bash
module load gcc/7.4.0 cuda/10.2 openmpi/4.0.1-gcc_7.4.0 cmake/3.17.3
```
(The default build for `parthenon` requires Python3 loaded. This is
not the case for `phoebus`.)

If you want output, make sure `hdf5` is available as well. On
`Darwin`, this likely means a spack environment. On `snow`, just load
the appropriate module. If you are compiling with mpi, use
`hdf5-parallel`, otherwise `hdf5-serial`.

If `cmake` can't find your hdf5 installation, You may have luck
setting `-DHDF5_ROOT=/path/to/hdf5` at configure time.

### Installing hdf5 through spack

If hdf5 isn't available on your target machine, here is one way to build it, using [Spack](https://spack.readthedocs.io/en/latest/getting_started.html).
```bash
# download spack
git clone https://github.com/spack/spack.git
# Activate the spack bash aliases. This typically needs to be done every time you want to use spack. I have it in my bashrc
. spack/share/spack/setup-env.sh
# load your modules
module load...
# Tell spack to learn about compilers and loaded modules
spack compiler find
spack external find
# Install parallel hdf5 with high-level libraries
spack install hdf5+mpi+hl
# load it
spack load hdf5
```

Installing from source, or just building without hdf5 will also both probably work.

## Example builds

Below are some example build commands

### MPI-parallel only

The following will build `phoebus` with MPI parallelism but no shared memory parallelism
```bash
cmake ..
make -j
```

### OpenMP-parallel

The following will build `phoebus` with OpenMP parallelism only
```bash
cmake -DPHOEBUS_ENABLE_MPI=Off -DPHOEBUS_ENABLE_OPENMP=ON ..
make -j
```

### Cuda

The following will build `phoebus` with no MPI or OpenMP parallelism.
```bash
cmake -DPHOEBUS_ENABLE_CUDA=On -DCMAKE_CXX_COMPILER=${HOME}/phoebus/external/singularity-eos/utils/kokkos/bin/nvcc_wrapper -DKokkos_ARCH_HSW=ON -DKokkos_ARCH_VOLTA70=ON -DPHOEBUS_ENABLE_MPI=OFF ..
```
A few notes for this one:
- Note here the `-DCMAKE_CXX_COMPILER` flag. This is necessary. You *must* set the compiler to `nvcc_wrapper` provided by Kokkos.
- Note the `-DKokkos_ARCH_*` flags. Those set the host and device microarchitectures and are required. The choice here is on an `x86_64` machine with a `volta` GPU.

## Build Options

The build options explicitly provided by `phoebus` are:

| Option             | Default | Comment                                            |
| ------------------ | ------- | -------------------------------------------------- |
| PHOEBUS_ENABLE_CUDA   | OFF     | Cuda                                               |
| PHOEBUS_ENABLE_HDF5   | ON      | HDF5. Required for output and restarts.            |
| PHOEBUS_ENABLE_MPI    | ON      | MPI. Required for distributed memory parallelism.  |
| PHOEBUS_ENABLE_OPENMP | OFF     | OpenMP. Required for shared memory parallelism.    |
| MACHINE_CFG        | None    | Machine-specific config file, optional.            |

Some relevant settings from Parthenon and Kokkos you may need to play with are:

| Option                   | Default | Comment                                                   |
| ------------------------ | ------- | --------------------------------------------------------- |
| CMAKE_CXX_COMPILER       | None    | Must be set to `nvcc_wrapper` with cuda backend           |
| Kokkos_ARCH_XXXX         | OFF     | You must set the GPU architecture when compiling for Cuda |

You can see all the Parthenon build options [here](https://github.com/lanl/parthenon/blob/develop/docs/building.md) and all the Kokkos build options [here](https://github.com/kokkos/kokkos/wiki/Compiling)

## Cmake machine configs

If you are proficient with `cmake` You can optionally write a `cmake`
file that sets the configure parameters that you like on a given
machine. Both `phoebus` and `parthenon` can make use of it. You can point to the file with
```
-DMACHINE_CFG=path/to/machine/file
```
at config time or by setting the environment variable `MACHINE_CFG` to point at it, e.g.,
```bash
export MACHINE_CFG=path/to/machine/file
```

An example machine file might look like
```
# Machine file for x86_64-volta on Darwin
message(STATUS "Loading machine configuration for Darwin x86-volta node")
message(STATUS "Assumes: module load module load gcc/7.4.0 cuda/10.2 openmpi/4.0.3-gcc_7.4.0 anaconda/Anaconda3.2019.10 cmake && spack load hdf5")
message(STATUS "Also assumes you have a valid spack installation loaded.")

set(PHOEBUS_ENABLE_CUDA ON CACHE BOOL "Cuda backend")
set(PHOEBUS_ENABLE_MPI OFF CACHE BOOL "No MPI")
set(Kokkos_ARCH_HSW ON CACHE BOOL "Haswell target")
set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "volta target")
set(CMAKE_CXX_COMPILER /home/jonahm/phoebus/external/parthenon/external/Kokkos/bin/nvcc_wrapper CACHE STRING "nvcc wrapper")
```

you could then configure and compile as

```bash
cmake -DMACHINE_CFG=path/to/machine/file ..
make -j
```

# Running

Run phoebus from the `build` directory as
```bash
./src/phoebus -i path/to/input/file.pin
```
The input files are in `phoebus/inputs/*`. There's typically one input file per problem setup file.

## Submodules

- `parthenon` asynchronous tasking and block-AMR infrastructure
- `singularity-eos` provides performance-portable equations of state and PTE solvers
- `singularity-opac` provides performance-portable opacities and emissivities
- `Kokkos` provides performance portable shared-memory parallelism. It allows our loops to be
  CUDA, OpenMP, or something else. By default we use the `Kokkos` shipped with `parthenon`.

## External (Required)

- `cmake` for building

## Optional

- `hdf5` for output
- `MPI` for distributed memory parallelism
- `python3` for visualization

## Clang-format

We use clang format for code cleanliness. The current version of
`clang-format` is pinned to `clang-format-12`, however, you can try
formatting with other versions at your own risk.

You can format all files in git history automatically by calling
`scripts/bash/format.sh` from anywhere within the Phoebus repo (but
outside of submodules).

You can specify which clang-format you use in the script with the CFM
environment variable. For example:
```bash
CFM=clang-format-12 ./scripts/bash/format.sh
```

# Contribute
We are always happy to have users contribute to `phoebus`.
To contribute, issue a pull request against the main branch. 
For more details on how to contribute to `phoebus`, please see [CONTRIBUTING.md](CONTRIBUTING.md).
We adhere to a [code of conduct](CODE_OF_CONDUCT.md).

# Copyright

© 2021-2024. Triad National Security, LLC. All rights reserved.  This
program was produced under U.S. Government contract 89233218CNA000001
for Los Alamos National Laboratory (LANL), which is operated by Triad
National Security, LLC for the U.S.  Department of Energy/National
Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of
Energy/National Nuclear Security Administration. The Government is
granted for itself and others acting on its behalf a nonexclusive,
paid-up, irrevocable worldwide license in this material to reproduce,
prepare derivative works, distribute copies to the public, perform
publicly and display publicly, and to permit others to do so.
