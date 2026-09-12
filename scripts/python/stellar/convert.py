
# refactor of the original convert.py that allows for conversion of phoebus profiles
# as a numpy array (instead of extra i/o with files), and also creates an output
# file with useful diagnostics/metrics about the model to help with *.pin file
# creation and provide reasonable floors/ceilings.
#
# law. 30 jun 2026


import os
import numpy as np
from astropy import constants as const

from seos import get_eos_bounds

# useful constants, in cgs
G = const.G.cgs.value
c = const.c.cgs.value
kB = const.k_B.cgs.value
msun = const.M_sun.cgs.value

# TODO: add enums to support both pre- and post-ADM column indices/labels for code readability (law).

def convert_PHB_profile(prof: np.ndarray):

    r'''
    Converts a post-ADM stellar/ccsne profile to a ``phoebus``-readable input file in code (phb) units. 
    Also calculates and provides the central density, along with characteristic mass and radius as given by

    - $M_0 = \rho_c^{-1/2} c^3 G^{-3/2}$
    - $R_0 = G M_0 c^{-2}$

    Args:
        prof (np.ndarray): Post-ADM, Eulerian stellar/ccsne profile (from MESA, KEPLER, GR1D...). 
            Assumed to be in following column order: [radius, density, temperature, ye, sie, velocity, pressure, density (adm), momentum (adm), $S$ (adm), $S^r_r$ (adm)]

    Returns:
        tuple:
        
        - prof (np.ndarray): The original profile with all primitive and ADM quantities converted to code units.
        - rhoc (float): Central density of the profile.
        - M0 (float): Characteristic mass of the profile.
        - R0 (float): Characteristic radius of the profile.
        - T0 (float): Characteristic time of the profile.
    '''

    # the input profile has the following columns:
    # 'radius [cm]', 'density [g/cm^3]', 'temperature [K]',  'ye', 'sie [erg/g]', 'velocity [cm/s]','pressure [dyne/cm^2]',  'density [ADM]', 'pressure [ADM]', 'S [ADM]', 'S_rr [ADM]'
    # we need to convert both the primitive and the ADM quantities

    prof_conv = np.copy(
        prof
    )  # making a fresh copy of the profile so we don't overwrite the physical units.

    rhoc = prof_conv[0, 1]  # central density
    M0 = 1.0 / (rhoc**0.5) * ((c**2.0 / G) ** 1.5)  # characteristic mass
    R0 = (G * M0) / (c**2.0)  # characteristic radius
    T0 = R0 / c # characteristic time

    # --- primitive quantities
    prof_conv[:, 0] *= (c**2.0) / G / M0  # radius
    prof_conv[:, 1] *= (G / c**2.0) ** 3.0 * M0**2.0  # mass density
    prof_conv[:, 2] *= kB  # temperature
    # note that we don't convert ye, it's unitless!
    prof_conv[:, 4] /= c**2.0  # specific internal energy
    prof_conv[:, 5] /= c  # velocity
    prof_conv[:, 6] *= G**3.0 / c**8.0 * M0**2.0  # pressure

    # --- ADM quantities
    prof_conv[:, 7] *= (G / c**2.0) ** 3.0 * M0**2.0  # adm density
    prof_conv[:, 8] *= G**3.0 / c**7.0 * M0**2.0  # adm momentum
    prof_conv[:, 9] *= G**3.0 / c**8.0 * M0**2.0  # adm entropy (?)
    prof_conv[:, 10] *= G**3.0 / c**8.0 * M0**2.0  # adm entropy, radial component (?)

    return prof_conv, rhoc, M0, R0, T0


def make_info_file( rhoc: float, M0: float, R0: float, T0: float, prof_conv: np.ndarray, model_name: str, model_type: str, EOSPATH: str, eos_type: str ='stellarcollapse', OUTPATH: str ='') -> None:
    '''
    Generates and outputs a summary/info file for a post-ADM, post unit conversion stellar/ccsne progenitor profile, including:

    - characteristic values used to convert to code units, 
    - useful length scales for the core-collapse problem, 
    - bounds of the progenitor profile and equation of state, and
    - conversion factors to convert between cgs and code (phb) units.

    This file is ASCII- and numpy-readable for ease of use when conducting post-simulation data analysis, e.g. ``np.loadtxt('s15.0_mesa.info', usecols=0)``

    File naming convention is `(model_name)_(model_type).info`.

    Args:
        rhoc (float): Central density of the profile.
        M0 (float): Characteristic mass of the profile.
        R0 (float): Characteristic radius of the profile.
        T0 (float): Characteristic time of the profile.
        prof_conv (np.ndarray): Post-ADM, converted Eulerian stellar/ccsne profile.
        model_name (str): Model name.
        model_type (str): Model type (e.g. MESA, KEPLER, GR1D...).
        EOSPATH (str): Directory that contains the *.h5 tabulated equation of state.
        eos_type (str, optional): The type of tabulated EOS. Options are 'StellarCollapse' or 'Helmholtz'; default is 'StellarCollapse'
        OUTPATH (str, optional): Desired directory to save file(s) in. Default is current working directory. 
    '''
    
    try:
        fout = open(
            os.path.join(OUTPATH, f'{model_name}_{model_type.lower()}.info'), 'w'
        )
        fmt = '%18.13e    %-25s\n'

        fout.write(
            f'# -----info, units, and conversions from {model_type.upper()} progenitor `{model_name}.`'
        )
        fout.write('\n\n')

        # --- characteristic quantities, length scales
        fout.write('# -----characteristic values [cgs]\n')
        fout.write(fmt % (rhoc, 'central density'))
        fout.write(fmt % (M0, 'characteristic mass'))
        fout.write(fmt % (R0, 'characteristic radius'))
        fout.write(fmt % (T0, 'characteristic time'))
        fout.write('\n')
        fout.write('# -----length scales [phb]\n')
        fout.write(fmt % (500e5 / R0, 'radius, 500 km'))
        fout.write(fmt % (1e9 / R0, 'radius, 1e4 km'))
        fout.write('\n')
        fout.write('# -----temporal scales [phb]\n')
        fout.write(fmt % (1e-3 / T0, 'time, 1 ms'))
        fout.write(fmt % (1.0 / T0, 'time, 1 s'))
        fout.write('\n')

        # --- model bounds
        fout.write('# -----bounds, progenitor [phb]\n')
        fout.write(fmt % (np.min(prof_conv[:, 1]), 'minimum density'))
        fout.write(fmt % (np.max(prof_conv[:, 1]), 'maximum density'))
        fout.write(fmt % (np.min(prof_conv[:, 2]), 'minimum temperature'))
        fout.write(fmt % (np.max(prof_conv[:, 2]), 'maximum temperature'))
        fout.write(fmt % (np.min(prof_conv[:, 4]), 'minimum sie'))
        fout.write(fmt % (np.max(prof_conv[:, 4]), 'maximum sie'))
        fout.write('\n')

        # --- eos bounds
        bounds = get_eos_bounds(EOSPATH, eos_type)

        fout.write('# -----bounds, equation of state [phb]\n')
        fout.write(fmt % (bounds[0] / (1 / ((c**2.0) / G / M0)), 'minimum density'))
        fout.write(fmt % (bounds[1] / (1 / ((c**2.0) / G / M0)), 'maximum density'))
        fout.write(fmt % (bounds[2] * kB, 'minimum temperature'))
        fout.write(fmt % (bounds[3] * kB, 'maximum temperature'))
        fout.write(fmt % (bounds[4] / (c**2.0), 'minimum sie'))
        fout.write(fmt % (bounds[5] / (c**2.0), 'maximum sie'))
        fout.write(fmt % (bounds[6], 'minimum ye'))
        fout.write(fmt % (bounds[7], 'maximum ye'))
        fout.write('\n')

        # --- conversion factors
        fout.write(
            '# -----conversion factors, multiply to get back to cgs [phb -> cgs]\n'
        )
        fout.write(fmt % (R0, 'radius'))
        fout.write(fmt % (1 / ((c**2.0) / G / M0), 'density'))
        fout.write(fmt % (1 / kB, 'temperature'))
        fout.write(fmt % (c**2.0, 'sie'))
        fout.write(fmt % (c, 'velocity'))
        fout.write(fmt % (1 / (G**3.0 / c**8.0 * M0**2.0), 'pressure'))
        fout.write(fmt % (T0, 'time'))
        fout.write('\n')

        fout.close()

    except:
        raise IOError(
            f'unable to write summary file for {model_type.upper()} model `{model_name}`.'
        )
