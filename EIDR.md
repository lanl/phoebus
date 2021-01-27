Phoebus Request for Open-Sourcing Description
===

# Description

Phoebus is a performance portable code for simulating compact object
astrophysics. This includes core-collapse supernovae, the in-spiral
and merger of neutron stars and black holes, and the accretion disks
formed in the aftermath of these cataclysmic events. In general, this
problem requires: General relativity and gravitational waves, neutrino
and weak physics, relativistic magnetohydrodynamics, and the ability
to use open-sourced tabulated equations of state and neutrino
opacities such as those provided by stellarcollapse.org.


## Details
 
The astrophysics methods and models implemented in Phoebus are
available in the open literature:

- Finite volumes are used to model the general relativistic
  magnetohydrodynamics. For a review of the equations used and
  thethods used to solve them, see Font. Living Rev. Relativ. 11, 7
  (2008). (https://doi.org/10.12942/lrr-2008-7)
- Monte Carlo Neutrino transport and Neutrino fluid coupling as
  described in Miller, Ryan and Dolence. ApJS, 241:2
  (2019). (https://doi.org/10.3847/1538-4365/ab09fc)
- The MOCMC method for neutrino transport, as described in Ryan and
  Dolence. ApJ 891 118
  (2020). (https://doi.org/10.3847/1538-4357/ab75e1)
- Dynamical spacetimes as described in (for example) Daszuta et al.,
  (2020). (https://arxiv.org/abs/2101.08289)
  
To enable performance portability and mesh refinement, Phoebus is
built on the open-source, performance-portable AMR library Parthenon
(available here https://github.com/lanl/parthenon).

External packages, described in the dependencies section, can be
linked into the code to enable microphysics, such as tabulated
equations of state, neutrino charged-current interactions, and in-line
r-process nuclesynthesis.

## Applications

Phoebus is purpose built to model compact object astrophysics,
including core collapse supernovae, binary black hole mergers, binary
neutron star mergers, and black hole neutron star mergers, and the
aftermaths of these events. It is capable of modeling situations where
general or special relativistic fliud dynamics or neutrino physics are
relevant.

# Development Status and History

## When

Phoebus is under active development starting in October 2020. So far
it has been developed primarily by Josh Dolence, Jonah Miller, and Ben
Ryan. Phoebus is expected to be a collaboration platform with other
laboratories such as LBNL and with vendors such as NVIDIA and we
expect more developers to join the effort in time as appropriate.

# Where

Phoebus is being developed at LANL.

## Mission or Purpose for which the Software Was Developed

Phoebus is being developed in support of the Next Generation Platforms
effort under the ASC progrem.

## Funding

Phoebus is funded by the Next Generation Platforms effort under the
ASC program.

# Software Details

## Operating System

Phoebus has only been tested on the Ubuntu, Debian, and Redhat flavors
of Linux. However, compilation on other unix systems should be
possible.

## Compilers

Phoebus has been tested with the GNU compiler and the NVIDIA nvcc Cuda
compiler. We intend to support a wide variety of compilers including
Intel, Clang, and IBM XL.

## Languages

Phoebus is written in C++ to be compatible with NVIDIA Cuda. We expect
some analysis code written in support of Phoebus to be written in
Python3.

## Dependencies

NO source code is included for any dependencies. Both static and
dynamic linking are supported.

- Phoebus optionally supports the HDF5 file format
  (https://www.hdfgroup.org/solutions/hdf5/) for file input/output.
- Phoebus optionally supports Catch2
  (https://github.com/catchorg/Catch2) for unit testing
- Analysis code requires the Python3 scientific python stack,
  including Python3, numpy, scipy, and matplotlib.
- Phoebus optionally supports Cuda
  (https://developer.nvidia.com/cuda-zone) for GPU support
- Phoebus requires Kokkos (https://github.com/kokkos/kokkos) for
  performance portability
- Phoebus requires the Parthenon library
  (https://github.com/lanl/parthenon) for mesh refinement
- Phoebus requires the singularity-eos and singularity-opac libraries
  to support tabulated microphysics data. These libraries will be
  disclosed in a separate EIDR.
- Phoebus optionally supports nuclear reaction networks such as SkyNet
  (https://doi.org/10.3847/1538-4365/aa94cb) to model r-process
  nucleosynthesis and heating rates.
