import numpy as np
import argparse
from pathlib import Path

c = 2.99792458e10  # speed of light in cm/s

def read_time_series(filename):
    """Reads a time series file with radial profiles at each time step."""
    time_series_data = {}
    ordered_times = []

    with open(filename, 'r') as file:
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

def find_nearest_time(target_time, times):
    """Find the time closest to the requested target time."""
    return min(times, key=lambda t: abs(t - target_time))

def save_snapshot_at_time(time, data_dicts, gr1d_output_dir):
    gr1d_output_dir = Path(gr1d_output_dir)
    gr1d_output_dir.mkdir(parents=True, exist_ok=True)

    r, rho_m = data_dicts['rho'][time]
    _, eps = data_dicts['eps'][time]
    _, temp = data_dicts['temp'][time]
    _, ye = data_dicts['ye'][time]
    _, p = data_dicts['p'][time]
    _, v = data_dicts['v'][time]

    v_ang = np.zeros_like(r)
    rho = rho_m + eps * rho_m / c**2

    np.save(gr1d_output_dir / 'r.npy', r)
    np.save(gr1d_output_dir / 'v.npy', v)
    np.save(gr1d_output_dir / 'v_ang.npy', v_ang)
    np.save(gr1d_output_dir / 'rho_m.npy', rho_m)
    np.save(gr1d_output_dir / 'eps.npy', eps)
    np.save(gr1d_output_dir / 'rho.npy', rho)
    np.save(gr1d_output_dir / 'temp.npy', temp)
    np.save(gr1d_output_dir / 'ye.npy', ye)
    np.save(gr1d_output_dir / 'p.npy', p)

    print(f"Saved snapshot at time {time:.6f} s to '{gr1d_output_dir}'")

def main(time, path='20M_convection_Mariam/', output_dir='gr1d_output_snapshot/'):
    rho_data, rho_times = read_time_series(Path(path) / 'rho.xg')
    eps_data, eps_times = read_time_series(Path(path) / 'eps.xg')  # assuming entropy.xg holds sie
    temp_data, temp_times = read_time_series(Path(path) / 'temperature.xg')
    ye_data, ye_times = read_time_series(Path(path) / 'ye.xg')
    p_data, p_times = read_time_series(Path(path) / 'press.xg')
    v_data, v_times = read_time_series(Path(path) / 'v.xg')

    # Ensure all time series contain the time
    times_common = set(rho_times) & set(eps_times) & set(temp_times) & set(ye_times) & set(p_times) & set(v_times)
    if not times_common:
        raise ValueError("No overlapping times in all files")

    nearest_time = find_nearest_time(time, sorted(times_common))

    data_dicts = {
        'rho': rho_data,
        'eps': eps_data,
        'temp': temp_data,
        'ye': ye_data,
        'p': p_data,
        'v': v_data
    }

    save_snapshot_at_time(nearest_time, data_dicts, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract snapshot from GR1D time series at a given time.")
    parser.add_argument('time', type=float, help='Target time in seconds  (NOTE: not a postbounce time)')
    parser.add_argument('--path', type=str, default='20M_convection_Mariam/', help='Path to data files')
    parser.add_argument('--output_dir', type=str, default='GR1D_output_snapshot/', help='Directory to save .npy outputs')

    args = parser.parse_args()
    main(args.time, path=args.path, output_dir=args.output_dir)
