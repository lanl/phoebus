FROM debian:bullseye-20190708-slim as dev-base
#FROM debian as dev-base
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

#RUN export HDF5_DIR=/usr/lib/aarch64-linux-gnu/hdf5/serial/
#RUN apt-get install --fix-missing -y python3-pip python3-dev
#RUN pip3 install --upgrade pip setuptools wheel && pip3 install numpy scipy matplotlib ipython jupyter pandas sympy 
#RUN export HDF5_DIR=/usr/lib/aarch64-linux-gnu/hdf5/serial/ && export HDF5_MPI="OFF" && pip install --no-binary=h5py h5py

#RUN curl https://stellarcollapse.org/EOS/SLy4_3335_rho391_temp163_ye66.h5.bz2 -o /SLy4_3335_rho391_temp163_ye66.h5.bz2 && bunzip2 /SLy4_3335_rho391_temp163_ye66.h5.bz2 
