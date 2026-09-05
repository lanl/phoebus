
# new skeleton for a unified library that can handle retrieval and formatting 
# of progenitor data for a unified case; should manage profiles taken from 
# the following stellar evolution/ccsne codes:
#     > MESA
#     > KEPLER
#     > GR1D
# also contains some helper utilities for reading in GR1D time series data and bounce times.
#
# law. 16 jun 2026


from astropy import constants as const
import pandas as pd
import numpy as np
import os

# we won't need this after testing
from scipy.interpolate import CubicSpline as cubic


from convert import convert_PHB_profile, make_info_file


def get_MESA_profile( PATH: str, header: int = 4, verbose: bool = False ) -> np.ndarray:
 
    '''
    Retrieves a MESA stellar evolution profile.

    Args:
        PATH (str): Path to the desired profile.
        header (int, optional): Line number where column headers begin (zero-indexed). Default is 4.
        verbose (bool, optional): Enables command line output. Defaults to False.

    Returns:
        np.ndarray: original profile values of [radius, velocity, density, pressure, ye, temperature, energy, omega, abar, zbar], from innermost radial coordinate outward.
    '''

    # loading in the profile using pandas dataframe
    try: 
         prof = pd.read_csv(PATH, sep=r"\s+", header=header)
    except: 
         raise FileNotFoundError(f'MESA profile not found at {PATH}.')

    # reversing the profile so that our inner radial coord is at 0.
    prof = prof[::-1].reset_index(drop=True)

    # quick check to see if our model was rotating or not:
    try: 
         omega = prof['omega']
    except KeyError:
        omega = np.zeros( len(prof) )
        print('>>> angular velocity not found, setting to zero.')
        
    # new numpy array of needed quantities.
    profnp = np.column_stack([
                prof['radius_cm'],
                prof['velocity'],
                prof['density'],
                prof['pressure'],
                prof['ye'],
                prof['temperature'],
                prof['energy'],
                omega,
                prof['abar'],
                prof['zbar']
            ])
    
    return profnp


def get_KEPLER_profile( PATH: str, header: int = 1, verbose: bool = False ) -> np.ndarray:
 
    '''
    Retrieves a KEPLER-style stellar evolution profile.

    Args:
        PATH (str): Path to the desired profile.
        header (int, optional): Line number where column headers begin (zero-indexed). Default is 1.
        verbose (bool, optional): Enables command line output. Defaults to False.

    Returns:
        np.ndarray: original profile values of [radius, velocity, density, pressure, ye, temperature, energy, omega, abar, zbar], from innermost radial coordinate outward.
    '''

    # loading in the profile using pandas dataframe
    try: 
         prof = pd.read_csv(PATH, sep='  +', header=header, engine='python')
    except: 
         raise FileNotFoundError(f'KEPLER profile not found at {PATH}.')


    # quick check to see if our model was rotating or not:
    try: 
         omega = prof['cell angular velocity']
    except KeyError:
        omega = np.zeros( len(prof) )
        print('>>> angular velocity not found, setting to zero.')
        
    # new numpy array of needed quantities.
    profnp = np.column_stack([
                prof['cell outer radius'],
                prof['cell outer velocity'],
                prof['cell density'],
                prof['cell pressure'],
                prof['cell Y_e'],
                prof['cell temperature'],
                prof['cell specific energy'],
                omega,
                prof['cell A_bar'],
                prof['cell A_bar'] * prof['cell Y_e'] # since KEPLER doesn't provide zbar
            ])
    
    return profnp


def get_GR1D_profile( PATH: str, at_bounce: bool = True, time: float = -1.0, verbose: bool = False ) -> np.ndarray:
     
    '''
    Retrieves a GR1D explosion profile.

    Args:
        PATH (str): Path to the desired profile.
        at_bounce (bool, optional): Finds the time of core bounce from GR1D's `tbounce.dat` output file.  Defaults to True.
        time (float, optional): If at_bounce is false, use GR1D output files nearest to this time (in seconds).
        verbose (bool, optional): Enables command line output. Defaults to False.

    Returns:
        np.ndarray: original profile values of [radius, velocity, density, pressure, ye, temperature, energy, omega, abar, zbar], from innermost radial coordinate outward.
    '''


    try:

        if at_bounce: # pulls the bounce time from GR1D's default output
            time = get_tbounce( PATH )
    
        rho_data, rho_times     = read_time_series( os.path.join(PATH,'rho.xg') )
        eps_data, eps_times     = read_time_series( os.path.join(PATH,'eps.xg') )  # assuming entropy.xg holds sie
        temp_data, temp_times   = read_time_series( os.path.join(PATH,'temperature.xg') )
        ye_data, ye_times       = read_time_series( os.path.join(PATH,'ye.xg') )
        p_data, p_times         = read_time_series( os.path.join(PATH,'press.xg') )
        v_data, v_times         = read_time_series( os.path.join(PATH,'v.xg') )
        v_data, v_times         = read_time_series( os.path.join(PATH,'v.xg') )
        # TODO: i think the conversion of abar, zbar is incorrect here... fix later
        zbar_data, zbar_times   = read_time_series( os.path.join(PATH,'xzbar.xg') )
        abar_data, abar_times   = read_time_series( os.path.join(PATH,'xabar.xg') )

    except: 
         raise FileNotFoundError(f'GR1D *.xg, *.dat files not found at {PATH}.')

    try: 
        omg_data, omg_times   = read_time_series( os.path.join(PATH,'omega.xg') )
        has_omega = True
    except:
        has_omega = False
        print('>>> angular velocity not found, setting to zero.')
    
    if has_omega:  
        times_common = set(rho_times) & set(eps_times) & set(temp_times) & set(ye_times) & set(p_times) & set(v_times) & set(zbar_times) & set(abar_times) & set(omg_times)
    else:
        times_common = set(rho_times) & set(eps_times) & set(temp_times) & set(ye_times) & set(p_times) & set(v_times) & set(zbar_times) & set(abar_times)
    
    # Ensure all time series contain the time
    if not times_common: raise ValueError('no overlapping times in all files.')
    
    nearest_time = find_nearest_time( time, sorted(times_common) )

    radius, rho = rho_data[nearest_time]
    _, eps      = eps_data[nearest_time]
    _, temp     = temp_data[nearest_time]
    _, ye       = ye_data[nearest_time]
    _, press    = p_data[nearest_time]
    _, vel      = v_data[nearest_time]
    _, zbar     = zbar_data[nearest_time]
    _, abar     = abar_data[nearest_time]

    if has_omega: 
        _, omega = omg_data[nearest_time]
    else: 
        omega = np.zeros_like(radius)

    profnp = np.column_stack([
                radius,
                vel,
                rho,
                press,
                ye,
                temp / const.k_B.to('MeV/K').value, # converting back to kelvin
                eps,
                omega,
                abar,
                zbar
            ])
    
    return profnp, time


### ---------------------------------------------------------------
### ------------ data handling utilities for going from lagrangian-style grids (most 1d codes, mass coord) to eulerian resolution for phoebus

# takes a weighted moving average of data, intended for smoothing of velocity profiles after subsampling...
def wmavg( data: np.ndarray, n: int = 1 , exp: int = 1, use_inverse: bool = True, avg_edges: bool = True, verbose: bool = False):

    r'''
    Takes a weighted moving average of an array, using either a traditional or inverse method.
        
    1. $w_i = i^e$ (traditional)
    2. $w_i = \frac{1}{i^e}$ (inverse)

    for $i = 1...n$ and exponent $e$. Weights are applied to the data in the array from indices $(j - n) \ldots (j + n)$ so that the $n$ elements to the left and right of the central value at index j are weighted accordingly and applied to the average.
    The central value $a$ is weighted at one. So for some central value $a$, we find
        
    $$a_{wma} = \sum_{i = 1}^{n} x_i w_i \cdot (\sum_{i = 1}^{n} w_i)^{-1}$$

    Args:
        data (np.ndarray): The array to be averaged.
        n (int, optional): The averaging `window`. For some central index j, averaging will occur from indices $(j - n) \ldots (j + n)$. Default is 1.
        exp (int, optional): The exponent to raise each weight to. Default is 1.
        use_inverse (bool, optional): Enables the use of the inverse weighting method. Defaults to True.
        avg_edges (bool, optional): Enables a reduced weighted mean average on the values at indices $0 \ldots n$ and $(m - n) \ldots m$ for an array of size m. Default is True, 
            recommended to avoid edge effects or discontinuities in data.
        verbose (bool, optional): Enables command line output. Defaults to False.

    Returns:
        np.ndarray: The `smoothed` data with a rolling weighted mean average applied to every element.

    '''
    datavg = np.copy(data)

    # constructing our array of weights
    if use_inverse:
        ws = np.ones(n) / ((np.arange(n) + 2) ** exp) 
        weights = np.concatenate((ws[::-1], [1], ws), axis=0)
    else:
        ws = np.arange(n) + 1
        weights = np.concatenate((ws[::-1], [1], ws), axis=0)
    if verbose: print(weights)

    # applying weights to the data from indices n, i - n.    
    for i in range(n, data.size - n): 
            datavg[i] = np.sum(data[i - n: i + 1 + n] * weights) / np.sum(weights)

    # handling averaging from the edges inward (i.e. the first n zones and the last i - n zones)
    # if not enabled, you can get weird edge effects or discontinuities!
    if avg_edges:
        
        for i in range(n):
            datavg[i] = np.sum(data[: (2*i) + 1] * weights[n - i: n + i + 1]) / np.sum(weights[n - i: n + i + 1])
            datavg[-(i + 1)] = np.sum(data[-(2*i) - 1:] * weights[n - i: n + i + 1]) / np.sum(weights[n - i: n + i + 1])
            if verbose:
                print(i, data[: (2*i) + 1], weights[n - i: n + i + 1])
                print(i, data[-(2*i) - 1:], weights[n - i: n + i + 1])
        
    return datavg

def interp_to_eulerian( prof: np.ndarray, factor: int = 4, use_drad: bool = False, drad: float = 1e7, use_uniform: bool = False, unif_max: float = 5e9):

    '''
    Increases the radial resolution of a Lagrangian (mass coordinate) input profile to an Eulerian-appropriate radial resolution.

    Args:
        prof (np.ndarray): Original Lagrangian stellar/ccsne profile (from MESA, KEPLER, GR1D...).
        factor (int): Factor to increase radial resolution by. Default is 4, **not recommended** for ``phoebus``.
        use_drad (bool, optional): Enables the use of a specific radial spacing. Default is False.
        drad (float, optional): If use_drad enabled, specifices desired minimum radial spacing (in cm). Default is 1.0e7 cm
        use_uniform (bool, optional): Creates a uniformly spaced grid over the full radial domain. Default is False, **not recommended** for ``phoebus``.
        unif_max (float, optional): If use_uniform not enabled, sets the radius (in cm) at which the transiton from uniform (linear) radial grid to log radial grid occurs. Default is 5.0e9 cm.

    Returns:
        np.ndarray: The original profile, with increased radial resolution.

    '''

    nzones  = prof.shape[0]
    rad0     = prof[:, 0] # we assume the whole profile is passed in.
    drad0   = (rad0[-1] - rad0[0]) / (nzones * factor)

    # if we want to ensure a certain initial resolution in the profiles, e.g. dr = 1e7 cm
    if use_drad:
        while drad0 > drad:
            factor += 1
            drad0 = (rad0[-1] - rad0[0]) / (nzones * factor)
    
    if use_uniform:
        rad = np.linspace(rad0[0], rad0[-1], nzones * factor)
        prof_interp = np.zeros( (nzones * factor, prof.shape[1]) )
    
    else:
        unif_zones = np.abs(rad0 - unif_max).argmin()
        rad_unif = np.linspace(rad0[0], unif_max, unif_zones)
        rad_log = np.logspace(np.log10(unif_max), np.log10(rad0[-1]), 1000)
        rad = np.concatenate( (rad_unif, rad_log[1:]), axis = 0)
        prof_interp = np.zeros( (rad.size, prof.shape[1]) )

    
    prof_interp[:, 0] = rad

    for i in range(1, prof.shape[1]):
        prof_interp[:, i] = np.interp(rad, rad0, prof[:, i])

    return prof_interp


def fixup_rad_vel( prof: np.ndarray, verbose: bool = False ):

    r'''
    Applies a FLASH-style fixup (e.g. Couch et al. 2013...) to a stellar/ccsne profile. This includes:

    - Adjusting the radius to be cell-centered values (instead of edge)
    - Fixing the innermost zone of the velocity to be consistent with new radial values (all other primitives assumed piecewise const.)

    Args:
        prof (np.ndarray): Post-ADM, Eulerian stellar/ccsne profile (from MESA, KEPLER, GR1D...). 
            Assumed to be in following column order: [radius, density, temperature, ye, sie, velocity, pressure, density (adm), momentum (adm), $S$ (adm), $S^r_r$ (adm)]
        verbose (bool, optional): Enables command line output. Defaults to False.

    '''

    if verbose: print('>>> applying FLASH-style fixup to radius and first zone of velocity.')
        
    rad0 = prof[:, 0]
    vel = prof[:, 5]
    rad = np.zeros(rad0.size)
    
    # changing the radius to cell-centered values (rather than edge)
    rad[0] = 0.5 * rad0[0]
    drad = rad[1] - rad[0]

    for i in range(1, rad.size):
        rad[i] = 0.5 * ( rad0[i] + rad0[i - 1] )

    # fixing first zone of velocity profile
    dvdr = vel[0]/drad
    vel[0] = rad[0] * dvdr 

    # replacing with new radius
    prof[:, 0] = rad 


# KEEP FOR GENERAL TESTING/V&V!!
# old method of sampling
def subsample_depr( x, radius_cm, factor=4, verbose=False, get_factor=False, uniform=False):
    nzones = x.size # number of zones within the mesa raw data

    if uniform: 
        total_radius = radius_cm[-1] - radius_cm[0] # total radial domain of the mesa profile
        dr = total_radius / (nzones * factor) # << new uniform radiial spacing
        
        if get_factor: 
            while dr > 1e7:
                factor += 1
                dr = total_radius / (nzones * factor)

        if verbose:
            print(f'new radial spacing:\t{dr:1.4e}')
            print(f'zone factor:\t{factor}')

        new_grid = np.linspace(radius_cm[0], radius_cm[-1], nzones * factor)
    
    else:
        unif_radius = 5e9 - radius_cm[0] 
        unif_nzones = int(unif_radius / 2e7) # ensuring high resolution within the core to capture < 5e9 cm
        log_nzones = int(nzones * 0.70 * factor) # assuming 70% of the zones don't capture the core, multiplying by the general factor
        
        unif_grid = np.linspace(radius_cm[0], 5e9, unif_nzones)
        log_grid = np.logspace( np.log10(5e9), np.log10(radius_cm[-1]), log_nzones)
        new_grid = np.concatenate((unif_grid, log_grid[1:]), axis=0)

        if verbose:
            print(f'new uniform spacing:\t{unif_radius / unif_nzones :1.4e}')
            print(f'new log spacing:\t{np.log10( log_grid[1] - log_grid[0]):1.4e}')
            print(f'zone factor (log):\t{factor}')

    cinterp = cubic( radius_cm, x, extrapolate = None)
    
    if get_factor:
        return cinterp(new_grid), factor
    return cinterp(new_grid)


### ---------------------------------------------------------------
### ------------ handles saving profiles to either adm.py input or phoebus style input (in physical and code units)

def save_raw_profile( prof: np.ndarray, model_name: str, model_type: str, time: float = 0.0, OUTPATH: str = '', verbose: bool = False ):
    
    '''
    Saves a processed (pre-ADM) stellar/ccsne profile to a ASCII- and numpy-readable file (with nice formatting).
        File naming convention is `(model_name)_(model_type).prof`.
    
    Args:
         prof (np.ndarray): Post-ADM, Eulerian stellar/ccsne profile (from MESA, KEPLER, GR1D...). 
            Assumed to be in following column order: [radius, velocity, density, pressure, ye, temperature, energy, omega, abar, zbar]
        model_name (str): Model name.
        model_type (str): Model type (e.g. MESA, KEPLER, GR1D...).
        time (float, optional): Time elapsed since infall (in seconds). Default is 0 s, **recommended** for GR1D models.
        OUTPATH (str, optional): Desired directory to save file in. Default is current working directory.
        verbose (bool, optional): Enables command line output. Defaults to False.
        
    '''

    # formatting for the header (to align with columns)
    fmt_header = '\t'.join(['%-20s'] * 10)
    tup_header = ( 'radius [cm]', 'velocity [cm/s]', 'density [g/cm^3]', 'pressure [dyne/cm^2]', 'ye', 'temperature [K]', 'sie [erg/g]', 'omega [rad/s]', 'abar', 'zbar')

    np.savetxt(
        os.path.join(OUTPATH, f'{model_name}_{model_type.lower()}.prof'),
        prof,
        delimiter   ='\t',
        fmt         = '%20.15e',
        header      = f'initial {model_type.upper()} profile from model `{model_name}` at time {time:.4f} s.\n{fmt_header % tup_header}'
    )

    if verbose: print(f'>>> saved raw profile from {model_type.upper()} model `{model_name}` at time {time:.4f} s to {OUTPATH}.')


def save_ADM_profile( prof: np.ndarray, model_name: str, model_type: str, EOSPATH: str, eos_type: str = 'stellarcollapse', time: float = 0.0, OUTPATH: str = '', save_unconverted: bool = True, save_info: bool = True, verbose: bool = False ):

    r'''
    Saves a processed, post-ADM, converted (code units) ``phoebus`` input profile to a ASCII- and numpy-readable file (with nice formatting). Also saves:
       
    - unconverted (cgs units) ``phoebus`` input profile (optional)
    - model info file with characteristic values, unit conversions, and progenitor + eos bounds (optional)

    File naming conventions are:
    
    - `(model_name)_(model_type)_adm_converted.prof`.
    - `(model_name)_(model_type)_adm.prof`.
    - `(model_name)_(model_type).info`.

    Args:
         prof (np.ndarray): Post-ADM, Eulerian stellar/ccsne profile (from MESA, KEPLER, GR1D...). 
            Assumed to be in following column order: [radius, density, temperature, ye, sie, velocity, pressure, density (adm), momentum (adm), $S$ (adm), $S^r_r$ (adm)]
        model_name (str): Model name.
        model_type (str): Model type (e.g. MESA, KEPLER, GR1D...).
        time (float, optional): Time elapsed since infall (in seconds). Default is 0 s, **recommended** for GR1D models.
        OUTPATH (str, optional): Desired directory to save file(s) in. Default is current working directory.
        save_unconverted (bool, optional): If enabled, saves profile in cgs units. Defaults to True.
        save_info (bool, optional): If enabled, saves info file with characteristic values, conversions, and bounds. Defaults to True.
        verbose (bool, optional): Enables command line output. Defaults to False.
        
    '''

    # formatting for header
    fmt_header = '\t'.join(['%-20s'] * 11)
    tup_header = ( 'radius [cm]', 'density [g/cm^3]', 'temperature [K]',  'ye', 'sie [erg/g]', 'velocity [cm/s]','pressure [dyne/cm^2]', 'density [ADM]', 'momentum [ADM]', 'S [ADM]', 'S^r_r [ADM]')
    tup_header_conv = ( 'radius', 'density', 'temperature',  'ye', 'sie', 'velocity','pressure', 'density [ADM]', 'momentum [ADM]', 'S [ADM]', 'S^r_r [ADM]')
    
    # saves the unconverted profile with primitives + ADM quantities
    if save_unconverted:
        np.savetxt(
            os.path.join(OUTPATH, f'{model_name}_{model_type.lower()}_adm.prof'),
            prof,
            delimiter   ='\t',
            fmt         = '%20.15e',
            header      = f'primitives + ADM for {model_type.upper()} profile from model `{model_name}`at time {time:.4f} s.\n{fmt_header % tup_header}'
        )
        if verbose: print(f'>>> saved primitive + ADM profile from {model_type.upper()} model `{model_name}` at time {time:.4f} s to {OUTPATH}.')


    # converting the actual profile to phoebus code units
    prof_conv, rhoc, M0, R0, T0 = convert_PHB_profile( prof )

    # creates a summary/info file with useful conversions and progenitor bounds
    if save_info:
        make_info_file( rhoc, M0, R0, T0, prof_conv, model_name, model_type, EOSPATH, eos_type, OUTPATH)

    # saves the converted profile
    np.savetxt(
        os.path.join(OUTPATH, f'{model_name}_{model_type.lower()}_adm_converted.prof'),
        prof_conv,
        delimiter   ='\t',
        fmt         = '%20.15e',
        header      = f'converted primitives + ADM for {model_type.upper()} profile from model `{model_name}` at time {time:.4f} s.\n{fmt_header % tup_header_conv}'
    )
    if verbose: print(f'>>> saved converted primitive + ADM profile from {model_type.upper()} model `{model_name}` at time {time:.4f} s to {OUTPATH}.')


### ---------------------------------------------------------------
### ------------ GR1D utilities for reading time series data/output

def read_time_series( PATH: str ):
    '''
    Reads a time series file from GR1D (*.xg) for some arbitrary quantity at each time step.
    
    Args:
        PATH (str): Path to the run directory (should contain all GR1D output files).

    Returns:
        tuple:
    
        - data (dict): Time series data (key: time, value: quantity at that timestep).
        - times (list): Times that correspond to the keys in the full dictionary.
    
    '''
    time_series_data = {}
    ordered_times = []

    with open(PATH, 'r') as file:
        current_time = None
        r_values, quantity_values = [], []

        for line in file:
            line = line.strip()

            if line.startswith('"Time'):
                if current_time is not None and r_values:
                    time_series_data[current_time] = (np.array(r_values), np.array(quantity_values))
                try:
                    current_time = float(line.split('=')[1].strip())
                    ordered_times.append(current_time)
                except ValueError:
                    continue
                r_values, quantity_values = [], []
            else:
                try:
                    r, quantity = map(float, line.split())
                    r_values.append(r)
                    quantity_values.append(quantity)
                except ValueError:
                    continue

        if current_time is not None and r_values:
            time_series_data[current_time] = (np.array(r_values), np.array(quantity_values))

    return time_series_data, sorted(ordered_times)

def find_nearest_time( target_time: float, times: set ):
    '''
    Finds the time closest to the requested target time given a list of timesteps.
    
    Args:
        target_time (float): Desired time (in seconds) to search for.
        times (set): List or set of timesteps or keys to search through.
        
    Returns:
        float: Time or key nearest to the provided target time.

    '''
    return min(times, key=lambda t: abs(t - target_time))

def get_tbounce( PATH: str ) -> float:
    ''' 
    Retrieves the time of bounce from the default GR1D output.
    
    Args:
        PATH (str): Path to the run directory (should contain all GR1D output files).
        
    Returns:
        float: The time of core bounce, as given in `tbounce.dat`.
        
    '''
    return np.loadtxt( os.path.join(PATH, 'tbounce.dat') )[0]
