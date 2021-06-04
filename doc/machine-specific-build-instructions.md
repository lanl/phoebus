Machine-Specific Build Instructions
===

# Capulin

The following builds an `OpenMP`-parallel build for the ARM machine Capulin.

## Getting the code on to Capulin

I only have access to Capulin on the Turquoise system, which isn't
connected to the internet. So I had to get the source code there in a
crude way. I cloned it to my desktop with

```bash
git clone --recursive git@github.com:lanl/phoebus.git
```
and then copied it to Capulin via scp. To make this easier, I deleted the git archives in `phoebus` and its submodules and made a tarball. Tarball available on request.

## Modules, dependencies, etc

Capulin is a cray system, so it uses the Cray development environment. Spack is dodgy. Here are the modules I loaded:
```bash
module load cmake
module load cce
module load cray-mpich
module load cray-hdf5-parallel
module load cray-python
```

## MPI-only build

To build and run:
```bash
mkdir -p phoebus/bin
cd phoebus/bin
cmake -DPARTHENON_DISABLE_HDF5_COMPRESSION=ON ..
make -j
mpiexec -n 1 ./src/phoebus -i ../inputs/shocktube.pin
```

## MPI+OpenMP

To build and run:
```bash
mkdir -p phoebus/bin
cd phoebus/bin
cmake -DPARTHENON_DISABLE_HDF5_COMPRESSION=ON -DPHOEBUS_ENABLE_OPENMP=ON -DPAR_LOOP_LAYOUT=MDRANGE_LOOP -DPAR_LOOP_INNER_LAYOUT=TVR_INNER_LOOP ..
make -j
mpiexec -n 1 ./src/phoebus -i ../inputs/shocktube.pin
```

The arguments `-DPAR_LOOP_LAYOUT` and `-DPAR_LOOP_INNER_LAYOUT` are important here and have a significant effect on performance. The relevant options for `PAR_LOOP_LAYOUT` are:
- `MANUAL1D_LOOP`
- `MDRANGE_LOOP`
- `TPTTR_LOOP`
- `TPTVR_LOOP`
- `TPTTRTVR_LOOP`

The relevant options for `PAR_LOOP_INNER_LAYOUT` are
- `SIMDFOR_INNER_LOOP`
- `TVR_INNER_LOOP`

For more details, see arXiv:1905.04341. Note **do not** set `PAR_LOOP_LAYOUT=SIMDFOR_LOOP` as this will cause the code to segfault.

## MPI+pthreads

To buld and run:
```bash
mkdir -p phoebus/bin
cd phoebus/bin
cmake -DPARTHENON_DISABLE_HDF5_COMPRESSION=ON -DKokkos_ENABLE_PTHREAD=ON ..
make -j
mpiexec -n 1 ./src/phoebus -i ../inputs/shocktube.pin
```

## Builds with HDF5, but without MPI

Produces HDF5 errors. Likely there's a bug in `cray-hdf5` serial.
