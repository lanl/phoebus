#!/usr/bin/env python

# © 2023. Triad National Security, LLC. All rights reserved.  This
# program was produced under U.S. Government contract
# 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
# is operated by Triad National Security, LLC for the U.S.  Department
# of Energy/National Nuclear Security Administration. All rights in
# the program are reserved by Triad National Security, LLC, and the
# U.S. Department of Energy/National Nuclear Security
# Administration. The Government is granted for itself and others
# acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works,
# distribute copies to the public, perform publicly and display
# publicly, and to permit others to do so.

import argparse
import numpy as np
import sys
import os
from phoedf import phoedf
from multiprocessing import Pool

def get_torus_radial_profiles(dfile):

    profiles = {}

    # Geometry
    gdet = dfile.gdet
    gcov = dfile.gcov

    # Set up size of 1D profiles in memory. Use max and min of global domain and minimum
    # dx1 across all blocks.
    x1Max = max(np.array(dfile.BlockBounds)[:,1])
    x1Min = min(np.array(dfile.BlockBounds)[:,0])
    dx1_profile = min(dfile.Dx1)
    Nx1_profile = round((x1Max - x1Min) / dx1_profile)
    r = np.zeros(Nx1_profile)
    for i in range(Nx1_profile):
        r[i] = x1Min + (0.5 + i) * dx1_profile

    # Simulation state
    rho = dfile.GetRho()
    ug = dfile.GetUg()
    Pg = dfile.GetPg()
    ucon = dfile.Getucon()
    ucov = dfile.Getucov()
    vpcon = dfile.GetVpcon()
    bsq = dfile.GetPm() * 2
    sigma = bsq / rho
    alpha = dfile.alpha
    Gamma = dfile.GetGamma()

    # Initialize profiles
    Volume = np.zeros(Nx1_profile) # For volume averages
    Mass = np.zeros(Nx1_profile) # For density-weighted volume averages
    #Mdot = np.zeros(Nx1_profile) # Total accretion rate
    F_M_in = np.zeros(Nx1_profile) # Inflow mass flux
    F_M_out = np.zeros(Nx1_profile) # Outflow mass flux
    F_Eg = np.zeros(Nx1_profile) # Gas (MHD) energy flux
    F_Pg = np.zeros(Nx1_profile) # Gas radial momentum flux
    F_Lg = np.zeros(Nx1_profile) # Gas angular momentum flux
    Pg_sadw = np.zeros(Nx1_profile) # SADW gas pressure
    Pm_sadw = np.zeros(Nx1_profile) # SADW magnetic pressure
    JetP_m = np.zeros(Nx1_profile) # MHD jet power (sigma > 1)
    JetP_g = np.zeros(Nx1_profile) # hydro jet power (sigma > 1)
    beta = np.zeros(Nx1_profile) # Plasma beta
    vr = np.zeros(Nx1_profile) # SADW radial velocity
    # TODO(BRR) Precalculate some azimuthal averages like <u u_g> for Bernoulli. Store
    # these 2D arrays as well?

    for b in range(dfile.NumBlocks):
        dx1 = dfile.Dx1[b]
        dx2 = dfile.Dx2[b]
        dx3 = dfile.Dx3[b]
        dA = dx2*dx3*gdet
        dV = dx1*dA

        for i in range(dfile.Nx1):
            # Get min and max indices for which this value fits
            assert dfile.MaxLevel == 0, "This does not support mesh refinement!"
            fine_i = 1 # TODO(BRR) placeholder
            x1Min_block = dfile.BlockBounds[b][0]
            ix1_min = round(x1Min_block / dx1_profile)

            # Bernoulli number to distinguish inflows and outflows
            Be = - ((rho[b,:,:,i]  - Gamma[b,:,:,i]*ug[b,:,:,i])* ucov[b,0,:,:,i]) / rho[b,:,:,i] - 1.

            for ii in range(fine_i):
                ip = i + ix1_min
                Volume[ip] += np.sum(dV[b,:,:,i])
                Mass[ip] += np.sum(dV[b,:,:,i]*rho[b,:,:,i])

                # SADW, <Pg/Pm> rather than <Pg>/<Pm>
                # TODO(BRR) provide <Pg>/<Pm>
                Pg_sadw[ip] += np.sum(dV[b,:,:,i]*rho[b,:,:,i] * Pg[b,:,:,i])
                Pm_sadw[ip] += np.sum(dV[b,:,:,i]*rho[b,:,:,i] * bsq[b,:,:,i] / 2.)

                for j in range(dfile.Nx2):
                    for k in range(dfile.Nx3):
                        Tmunu_concov = dfile.GetTmunu_concov(b, k, j, i)
                        F_Eg[ip] += dA[b,k,j,i] * Tmunu_concov[1, 0]
                        F_Pg[ip] += dA[b,k,j,i] * Tmunu_concov[1, 1]
                        F_Lg[ip] += dA[b,k,j,i] * Tmunu_concov[1, 3]
                        if Be[k,j] > 0:
                            F_M_out[ip] += dA[b,k,j,i]*rho[b,k,j,i]*ucon[b,1,k,j,i]
                        else:
                            F_M_in[ip] += dA[b,k,j,i]*rho[b,k,j,i]*ucon[b,1,k,j,i]
                        if sigma[b,k,j,i] > 1.:
                            JetP_m[ip] += dA[b,k,j,i]*Tmunu_concov[1,0]
                            JetP_g[ip] += dA[b,k,j,i]*rho[b,k,j,i]*ucon[b,1,k,j,i]

                vr[ip] += np.sum(dV[b,:,:,i] * vpcon[b,0,:,:,i] / Gamma[b,:,:,i])

    beta = Pg_sadw / Pm_sadw

    profiles['r'] = r
    profiles['Volume'] = Volume
    profiles['Mass'] = Mass
    profiles['F_M'] = F_M_in + F_M_out
    profiles['F_M_in'] = F_M_in
    profiles['F_M_out'] = F_M_out
    profiles['F_Eg'] = F_Eg
    profiles['F_Pg'] = F_Pg
    profiles['F_Lg'] = F_Lg
    profiles['beta'] = beta
    profiles['JetP_m'] = JetP_m
    profiles['JetP_g'] = JetP_g
    profiles['vr'] = vr

    return profiles

def write_torus_radial_profiles(profiles, in_filename, out_filename):
    with open(out_filename, "w") as profile_file:
        for key in profiles.keys():
            profile_file.write(key + "\n")
            for i in range(len(profiles[key])):
                profile_file.write(str(profiles[key][i]) + " ")
            profile_file.write("\n")

    return out_filename

def process_file(filename, overwrite):

    out_filename = filename[:-5] + '.profile'

    if not overwrite and os.path.exists(out_filename):
        print(f"{os.path.basename(out_filename)} already exists! Skipping...")
        return

    print(f"Opening file {os.path.basename(filename)}... ")
    dfile = phoedf(filename)
    print(f"File {os.path.basename(filename)} opened.")

    profiles = get_torus_radial_profiles(dfile)
    print(f"Created profiles for file {os.path.basename(filename)}")

    out_filename = write_torus_radial_profiles(profiles, in_filename, out_filename)
    print(f"Wrote profiles to file {os.path.basename(out_filename)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get radial profiles from torus dump")
    parser.add_argument(
        "filenames", type=str, nargs="+", help="Files to derive radial profiles for"
    )
    parser.add_argument(
        "--nproc", type=int, default=1, help="Number of processes to use"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Whether to overwrite existing outputs"
    )
    args = parser.parse_args()

    print("Phoebus GRMHD analysis script for generating radial profiles from dumps")
    print(f"  Number of files: {len(args.filenames)}")

    p = Pool(processes=args.nproc)
    from itertools import repeat
    p.starmap(
        process_file,
        zip(args.filenames, repeat(args.overwrite))
    )

    #with multiprocessing.Pool(processes = args.nproc) as pool:
    #    results = pool.map(process_file, args.filenames)



    #for filename in args.filenames:
    #    print(f"Opening file {os.path.basename(filename)}... ")
    #    dfile = phoedf(filename)
    #    print(f"File {os.path.basename(filename)} opened.")

    #    profiles = get_torus_radial_profiles(dfile)
    #    print(f"Created profiles for file {os.path.basename(filename)}")

    #    out_filename = write_torus_radial_profiles(profiles, filename)
    #    print(f"Wrote profiles to file {os.path.basename(out_name)}")

