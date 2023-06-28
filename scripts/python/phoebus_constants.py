# © 2022. Triad National Security, LLC. All rights reserved.  This
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

from math import pi

cgs = {}
cgs["KM"] = 1e5
cgs["CL"] = 2.99792458e10
cgs["QE"] = 4.80320680e-10
cgs["ME"] = 9.1093826e-28
cgs["MP"] = 1.67262171e-24
cgs["MN"] = 1.67492728e-24
cgs["HPL"] = 6.6260693e-27
cgs["HBAR"] = 1.0545717e-27
cgs["KBOL"] = 1.3806505e-16
cgs["GNEWT"] = 6.6742e-8
cgs["SIG"] = 5.670400e-5
cgs["AR"] = 7.5657e-15
cgs["THOMSON"] = 0.665245873e-24
cgs["JY"] = 1.0e-23
cgs["PC"] = 3.085678e18
cgs["AU"] = 1.49597870691e13
cgs["MSOLAR"] = 1.989e33
cgs["RSOLAR"] = 6.96e10
cgs["LSOLAR"] = 3.827e33
cgs["EV"] = 1.60217653e-12
cgs["MEV"] = 1.0e6 * cgs["EV"]
cgs["GEV"] = 1.0e9 * cgs["EV"]
cgs["K"] = 1.380648780669e-16
cgs["GK"] = 1.0e9 * cgs["K"]
cgs["GFERM"] = 1.435850814907447e-49
cgs["GA"] = -1.272323
cgs["S2THW"] = 0.222321
cgs["alphafs"] = 1.0 / 137.0
cgs["NUSIGMA0"] = (
    4.0
    * (cgs["GFERM"] ** 2)
    * ((cgs["ME"] * cgs["CL"] ** 2) ** 2)
    / (pi * (cgs["HBAR"] * cgs["CL"]) ** 4)
)

scalefree = {}
scalefree["KM"] = 1.0
scalefree["CL"] = 1.0
scalefree["QE"] = 1.0
scalefree["ME"] = 1.0
scalefree["MP"] = 1.0
scalefree["MN"] = 1.0
scalefree["HPL"] = 1.0
scalefree["HBAR"] = scalefree["HPL"] / (2.0 * pi)
scalefree["KBOL"] = 1.0
scalefree["GNEWT"] = 1.0
scalefree["SIG"] = 1.0
scalefree["AR"] = 1.0
scalefree["THOMSON"] = 1.0
scalefree["JY"] = 1.0
scalefree["PC"] = 1.0
scalefree["AU"] = 1.0
scalefree["MSOLAR"] = 1.0
scalefree["RSOLAR"] = 1.0
scalefree["LSOLAR"] = 1.0
scalefree["EV"] = 1.0
scalefree["MEV"] = 1.0e6 * scalefree["EV"]
scalefree["GEV"] = 1.0e9 * scalefree["EV"]
scalefree["K"] = 1.0
scalefree["GK"] = 1.0e9 * scalefree["K"]
scalefree["GFERM"] = 1.0
scalefree["GA"] = -1.0
scalefree["S2THW"] = 1.0
scalefree["alphafs"] = 1.0 / 137.0
scalefree["NUSIGMA0"] = (
    4.0
    * (scalefree["GFERM"] ** 2)
    * ((scalefree["ME"] * scalefree["CL"] ** 2) ** 2)
    / (pi * (scalefree["HBAR"] * scalefree["CL"]) ** 4)
)
