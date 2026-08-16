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
    
    nan_file_name = "nan_record_valid.npz"
    bound_file_name = 'bound_values_valid.npz'

    # initialize normalization values if not exist, else load them
    nan_file = {}
    bound_file = {}

    for var in list_single_vars + ['cmpa']:
        nan_file[var] = []
        bound_file[var] = []
    for var in list_pressure_vars:
        for level in DEFAULT_PRESSURE_LEVELS:
            nan_file[f'{var}_{level}'] = []
            bound_file[f'{var}_{level}'] = []

    file_paths = os.listdir(root_dir)
    file_paths = sorted(file_paths)

    for file in tqdm(file_paths, desc='hour', position=0):
        data_path = os.path.join(root_dir, file)
        data = get_data_given_path(data_path, VARS)

        for var in (list_single_vars + list_pressure_vars + ['cmpa']):
            if var in SINGLE_LEVEL_VARS + ['cmpa']:
                if (True in np.isnan(data['input'][var])):
                    nan_file[var].append(file)
                bound_file[var].append((np.abs(data['input'][var])).max())
            else:
                for i, level in enumerate(DEFAULT_PRESSURE_LEVELS):
                    if (True in np.isnan(data['input'][f'{var}_{level}'])):
                        nan_file[f'{var}_{level}'].append(file)
                    bound_file[f'{var}_{level}'].append((np.abs(data['input'][f'{var}_{level}'])).max())
    np.savez(os.path.join(save_dir, nan_file_name), **nan_file)
    np.savez(os.path.join(save_dir, bound_file_name), **bound_file)

if __name__ == "__main__":
    main()



