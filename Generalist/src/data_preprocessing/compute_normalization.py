import os
import argparse
import numpy as np
import xarray as xr
from tqdm import tqdm
import h5py
from typing import Any, Dict, Optional, Tuple, Sequence
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.data_utils import (
    CONSTANTS,
    SINGLE_LEVEL_VARS,
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS
)

# change as needed
VARS = [
    't2m',
    'ts',
    'q2m',
    'psl',
    'u10m',
    'v10m',
    'ps',
    'cmpa',

    'u',
    'v',
    't',
    'h',
    'Qv',
]
def get_data_given_path(
        path: str, variables: Sequence[str]
) -> Dict:
    """Prepare the data given the data path and variables.
    
    """
    with h5py.File(path, 'r') as f:
        data = {
            main_key: {
                sub_key: np.array(value) for sub_key, value in group.items()
        } for main_key, group in f.items() if main_key in ['input']}

    return data

def parse_args():
    parser = argparse.ArgumentParser(description='Regridding NetCDF files.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save regridded files.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    root_dir = args.root_dir
    save_dir = args.save_dir
    
    os.makedirs(save_dir, exist_ok=True)

    list_constant_vars = [v for v in VARS if v in CONSTANTS]
    list_single_vars = [v for v in VARS if v in SINGLE_LEVEL_VARS and v not in CONSTANTS]
    list_pressure_vars = [v for v in VARS if v in PRESSURE_LEVEL_VARS]
    
    mean_file_name = "normalize_mean.npz"
    std_file_name = "normalize_std.npz"

    # initialize normalization values if not exist, else load them
    normalize_mean = {}
    normalize_std = {}

    for var in list_single_vars + ['cmpa']:
        normalize_mean[var] = []
        normalize_std[var] = []
    for var in list_pressure_vars:
        for level in DEFAULT_PRESSURE_LEVELS:
            normalize_mean[f'{var}_{level}'] = []
            normalize_std[f'{var}_{level}'] = []

    file_paths = os.listdir(root_dir)
    file_paths = sorted(file_paths)

    for file in tqdm(file_paths, desc='hour', position=0):
        data_path = os.path.join(root_dir, file)
        data = get_data_given_path(data_path, VARS)

        for var in (list_single_vars + list_pressure_vars + ['cmpa']):
            if var in SINGLE_LEVEL_VARS:
                normalize_mean[var].append(data['input'][var].mean())
                normalize_std[var].append(data['input'][var].std())
            elif var in PRESSURE_LEVEL_VARS:
                for i, level in enumerate(DEFAULT_PRESSURE_LEVELS):
                    normalize_mean[f'{var}_{level}'].append(data['input'][f'{var}_{level}'].mean())
                    normalize_std[f'{var}_{level}'].append(data['input'][f'{var}_{level}'].std())
            else:
                if data['input']['exist_cmpa']:
                    normalize_mean[var].append(data['input'][var].mean())
                    normalize_std[var].append(data['input'][var].std())
                else:
                    print(f'file {file} does not exist cmpa')
                    continue

    for var in (list_single_vars + list_pressure_vars + ['cmpa']):        
        if var in SINGLE_LEVEL_VARS + ['cmpa']:
            mean_over_files, std_over_files = np.array(normalize_mean[var]), np.array(normalize_std[var])
            # var(X) = E[var(X|Y)] + var(E[X|Y])
            variance = (std_over_files**2).mean() + (mean_over_files**2).mean() - mean_over_files.mean()**2
            std = np.sqrt(variance)
            # E[X] = E[E[X|Y]]
            mean = mean_over_files.mean()
            normalize_mean[var] = mean.reshape([1])
            normalize_std[var] = std.reshape([1])
        
            np.savez(os.path.join(save_dir, mean_file_name), **normalize_mean)
            np.savez(os.path.join(save_dir, std_file_name), **normalize_std)
        else:
            for l in DEFAULT_PRESSURE_LEVELS:
                var_lev = f'{var}_{l}'
                mean_over_files, std_over_files = np.array(normalize_mean[var_lev]), np.array(normalize_std[var_lev])
                # var(X) = E[var(X|Y)] + var(E[X|Y])
                variance = (std_over_files**2).mean() + (mean_over_files**2).mean() - mean_over_files.mean()**2
                std = np.sqrt(variance)
                # E[X] = E[E[X|Y]]
                mean = mean_over_files.mean()
                normalize_mean[var_lev] = mean.reshape([1])
                normalize_std[var_lev] = std.reshape([1])
        
            np.savez(os.path.join(save_dir, mean_file_name), **normalize_mean)
            np.savez(os.path.join(save_dir, std_file_name), **normalize_std)

if __name__ == "__main__":
    main()