# © 2021. Triad National Security, LLC. All rights reserved.  This
# program was produced under U.S. Government contract
# 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
# is operated by Triad National Security, LLC for the U.S.
# Department of Energy/National Nuclear Security Administration. All
# rights in the program are reserved by Triad National Security, LLC,
# and the U.S. Department of Energy/National Nuclear Security
# Administration. The Government is granted for itself and others
# acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works,
# distribute copies to the public, perform publicly and display
# publicly, and to permit others to do so.

FROM debian:bullseye-20190708-slim as dev-base
LABEL author="Luke Roberts" 

#ENV HTTP_PROXY http://proxyout.lanl.gov:8080
#ENV HTTPS_PROXY http://proxyout.lanl.gov:8080

# Install the required libraries
ARG DEBIAN_FRONTEND=noninteractive
RUN  apt-get update && apt-get install -y \
    build-essential \
    gfortran \ 
    gdb \ 
    gdbserver \
    make \
    cmake \
    git \
    curl \
    bzip2 \
    libhdf5-dev \
    liblapack-dev \
    libboost-all-dev \  
    libgsl-dev \
    hdf5-tools \ 
    catch2 \
    mpich
RUN export HDF5_DIR=/usr/lib/aarch64-linux-gnu/hdf5/serial/
RUN apt-get install --fix-missing -y python3-pip python3-dev
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install numpy scipy matplotlib ipython jupyter pandas sympy
RUN pip3 install h5py

#RUN curl https://stellarcollapse.org/EOS/SLy4_3335_rho391_temp163_ye66.h5.bz2 -o /SLy4_3335_rho391_temp163_ye66.h5.bz2 && bunzip2 /SLy4_3335_rho391_temp163_ye66.h5.bz2 
