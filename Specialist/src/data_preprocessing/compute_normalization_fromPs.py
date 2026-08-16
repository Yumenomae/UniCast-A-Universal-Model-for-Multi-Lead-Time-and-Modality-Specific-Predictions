import os
import argparse
import numpy as np
import xarray as xr
from tqdm import tqdm
import h5py
from typing import Any, Dict, Optional, Tuple, Sequence
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.physics_components.air_column_vars_ps_refined import compute_air_column_variable_taking_geography
from src.utils.data_utils import DEFAULT_PRESSURE_LEVELS

variables=[
  'cmpa',
  't2m',
  'ts',
  'q2m',
  'psl',
  'u10m',
  'v10m',
  'ps',
  
  'u_1000',
  'u_925',
  'u_850',
  'u_700',
  'u_600',
  'u_500',
  'u_400',
  'u_300',
  'u_250',
  'u_200',
  'u_150',
  'u_100',
  'u_50',
  'v_1000',
  'v_925',
  'v_850',
  'v_700',
  'v_600',
  'v_500',
  'v_400',
  'v_300',
  'v_250',
  'v_200',
  'v_150',
  'v_100',
  'v_50',
  't_1000',
  't_925',
  't_850',
  't_700',
  't_600',
  't_500',
  't_400',
  't_300',
  't_250',
  't_200',
  't_150',
  't_100',
  't_50',
  'h_1000',
  'h_925',
  'h_850',
  'h_700',
  'h_600',
  'h_500',
  'h_400',
  'h_300',
  'h_250',
  'h_200',
  'h_150',
  'h_100',
  'h_50',
  'Qv_1000',
  'Qv_925',
  'Qv_850',
  'Qv_700',
  'Qv_600',
  'Qv_500',
  'Qv_400',
  'Qv_300',
  'Qv_250',
  'Qv_200',
  'Qv_150',
  'Qv_100',
  'Qv_50',
]

# change as needed
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

    x = [data['input'][v] for v in variables]
    return np.stack(x, axis=0), data['input']['exist_cmpa']

def get_target_path(
        data_dir: str, year: str, inp_file_idx: int, steps: int
) -> str:
    """get the target path according to the input data index and forward steps.
    
    :param year: current year
    :param inp_file_idx: file index of the input in the current year
    :param steps: number of steps forward
    """
    target_file_idx = inp_file_idx + steps
    target_path = os.path.join(data_dir, f'{year}_{target_file_idx:04}.h5')
    if not os.path.exists(target_path):
        for i in range(steps):
            target_file_idx = inp_file_idx + i
            target_path = os.path.join(data_dir, f'{year}_{target_file_idx:04}.h5')
            if os.path.exists(target_path):
                max_step_forward = i
        remaining_steps = steps - max_step_forward
        next_year = year + 1
        target_path = os.path.join(data_dir, f'{next_year}_{remaining_steps-1:04}.h5')
    return target_path

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
    
    # mean_file_name = "normalize_assist_mean_advection_02.npz"
    # std_file_name = "normalize_assist_std_advection_02.npz"

    # initialize normalization values if not exist, else load them
    normalize_mean = {}
    normalize_std = {}

    lat = np.load('/home-ssd/Users/gm_lhy/zhengjq/Ours/data/lat.npy')
    lon = np.load('/home-ssd/Users/gm_lhy/zhengjq/Ours/data/lon.npy')
    assist_compute = compute_air_column_variable_taking_geography(pressure_level=DEFAULT_PRESSURE_LEVELS,
                                                    lat=lat,
                                                    lon=lon,
                                                    a=0.79)
    
    for var in ['airColumn_h']:
        normalize_mean[var] = []
        normalize_std[var] = []

    file_paths = os.listdir(root_dir)
    file_paths = sorted(file_paths)

    for file in tqdm(file_paths[:-1], desc='hour', position=0):
        data_path = os.path.join(root_dir, file)
        data, _ = get_data_given_path(data_path, variables)
        data = torch.from_numpy(data).unsqueeze(0)
        assert data.shape == (1, 73, 441, 845)

        year, inp_file_idx = os.path.basename(data_path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)

        target_path = get_target_path(root_dir, year, inp_file_idx, steps=1)
        target, exist_cmpa = get_data_given_path(target_path, variables)
        target = torch.from_numpy(target).unsqueeze(0)
        assert target.shape == (1, 73, 441, 845)

        if exist_cmpa == False: continue

        # target_dict = assist_compute.pressure_level_reshape_denorm(target, None)
        # fute_Qv = target_dict['Qv']
        fute_R = target[:, 0]

        upper_vars_dict, surface_vars_dict = assist_compute.pressure_level_reshape_denorm(data, None)

        u, v, Qv, t = upper_vars_dict['u'], upper_vars_dict['v'], upper_vars_dict['Qv'], upper_vars_dict['t']
        u10m, v10m = surface_vars_dict['u10m'], surface_vars_dict['v10m']
        q2m, ps, t2m = surface_vars_dict['q2m'], surface_vars_dict['ps'], surface_vars_dict['t2m']

        # negative_int_press_div_refined = assist_compute.air_column_humidity_advection_integration(Qv, q2m,
        #                                           u, v, u10m, v10m,
        #                                           t, t2m,
        #                                           ps)
        # negative_int_press_div_old = assist_compute_old.air_column_humidity_advection_integration(Qv, u, v)

        VIWV_refined = assist_compute.air_column_humidity(Qv, q2m, ps)  

        normalize_mean['airColumn_h'].append(VIWV_refined.mean())
        normalize_std['airColumn_h'].append(VIWV_refined.std())

    for var in ['airColumn_h']:        
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

if __name__ == "__main__":
    main()