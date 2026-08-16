import os
import argparse
import numpy as np
import xarray as xr
import h5py
from tqdm import tqdm
from Ctloader import ModelDataReader
import xarray as xr
from Interpolation import regrid_CMPA
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.data_utils import (
    CONSTANTS,
    SINGLE_LEVEL_VARS,
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS
)

target_lon, target_lat = np.arange(97.06, 122.39, 0.03), np.arange(17.2, 30.41, 0.03)
target_lon_idx, target_lat_idx = slice(134, 979, 1), slice(240, 681, 1)
# change as needed
VARS = SINGLE_LEVEL_VARS + PRESSURE_LEVEL_VARS

def create_constant(root_dir, save_dir, ctl_path):
    ctl_data_path = os.path.join(root_dir, 'data', '2019', '2019010100', 'postvar001')
    os.makedirs(save_dir, exist_ok=True)
    reader = ModelDataReader(ctl_path)
    data = reader.get_var(ctl_data_path)
    zs = data['zs'].squeeze() # (1117, 1331)
    zs = zs[target_lat_idx, target_lon_idx] # (441, 845)
    zs = (zs - zs.mean()) / zs.std() # normalize
    np.save(os.path.join(save_dir, 'terrain.npy'), zs)

    dem_and_land = xr.open_dataset('./data/south_dem.nc')
    dem, land = dem_and_land['dem'].values, dem_and_land['land'].values # (442, 847)
    dem, land = dem[:-1, :-2], land[:-1, :-2] # (441, 845)
    dem = (dem - dem.mean()) / dem.std()
    np.save(os.path.join(save_dir, 'dem.npy'), dem)
    np.save(os.path.join(save_dir, 'lsm.npy'), land)

    landtype = xr.open_dataset('./data/south_landuse.nc')
    landtype = landtype['landuse'].values # (442, 847)
    landtype = landtype[:-1, :-2] # (441, 845)
    np.save(os.path.join(save_dir, 'landtype.npy'), landtype)

    lat = target_lat
    lat.sort()
    lon = target_lon
    lon.sort()
    np.save(os.path.join(save_dir, 'lat.npy'), lat)
    np.save(os.path.join(save_dir, 'lon.npy'), lon)

def create_one_step_dataset(root_dir, save_dir, ctl_path, split, years, list_vars):
    save_dir_split = os.path.join(save_dir, split)
    os.makedirs(save_dir_split, exist_ok=True)
    reader = ModelDataReader(ctl_path)
    
    list_constant_vars = [v for v in list_vars if v in CONSTANTS]
    list_single_vars = [v for v in list_vars if v in SINGLE_LEVEL_VARS]
    list_pressure_vars = [v for v in list_vars if v in PRESSURE_LEVEL_VARS]
    
    cmpa_prefix = 'R01H_FRT_'
    for year in tqdm(years, desc='years', position=0):
        ctl_data_dir = os.path.join(root_dir, 'data', str(year))
        file_paths = os.listdir(ctl_data_dir)
        file_paths = sorted(file_paths)

        cmpa_dir = os.path.join(root_dir, 'cmpa', '3km')
        cmpa_month = str(year) + '01'
        cmpa_data_dir = os.path.join(cmpa_dir, cmpa_month)

        idx_in_year = 0

        for file in tqdm(file_paths, desc='hour', position=1):

            data_dict = {
                    'input': {'time': file}
                }
            # ctl_data_dict = reader.get_var(os.path.join(ctl_data_dir, file, 'postvar001')) # for 2019-2023
            ctl_data_dict = reader.get_var(os.path.join(ctl_data_dir, file, 'postvar001.nc')) # for 2017-2018
            for var in (list_single_vars + list_pressure_vars):
                if var in list_single_vars:
                    data_dict['input'][var] = ctl_data_dict[var][0, target_lat_idx, target_lon_idx] # (441, 845)
                else:
                    ds_np = ctl_data_dict[var][:, target_lat_idx, target_lon_idx] # (13, 441, 845)
                    for i, level in enumerate(DEFAULT_PRESSURE_LEVELS):
                        data_dict['input'][f'{var}_{level}'] = ds_np[i, :]
                # data_dict['input'][var] = ctl_data_dict[var]

            if file[:6] != cmpa_month:
                cmpa_month = file[:6]
                cmpa_data_dir = os.path.join(cmpa_dir, cmpa_month)

            cmpa_file = os.path.join(cmpa_data_dir, cmpa_prefix + file + '.npy')
            if os.path.exists(cmpa_file):
                data_dict['input']['exist_cmpa'] = True
                cmpa = np.load(cmpa_file)
                cmpa = regrid_CMPA(cmpa) # (441, 845)
                data_dict['input']['cmpa'] = cmpa
            else:
                data_dict['input']['exist_cmpa'] = False
                data_dict['input']['cmpa'] = np.zeros((441, 845))
                print(f'{cmpa_file} does not exist!')
            
            with h5py.File(os.path.join(save_dir_split, f'{year}_{idx_in_year:04}.h5'), 'w', libver='latest') as f:
                for main_key, sub_dict in data_dict.items():
                    # Create a group for the main key (e.g., 'input' or 'output')
                    group = f.create_group(main_key)
                    
                    # Now, save each array in the sub-dictionary to this group
                    for sub_key, array in sub_dict.items():
                        if sub_key not in ['time', 'exist_cmpa']:
                            group.create_dataset(sub_key, data=array, compression=None, dtype=np.float32)
                        else:
                            group.create_dataset(sub_key, data=array, compression=None)
            
            idx_in_year += 1


def parse_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save regridded files.')
    parser.add_argument('--ctl_path', type=str, required=True, help='Ctl parser file to load ctl data.')
    parser.add_argument('--start_year', type=int, default=1979, help='Start year for the data range.')
    parser.add_argument('--end_year', type=int, default=2019, help='End year for the data range.')
    parser.add_argument("--split", type=str, default="train", help="Split of the dataset (train, val, test).")
    
    return parser.parse_args()


def main():
    args = parse_args()

    create_constant(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        ctl_path=args.ctl_path,
    )

    create_one_step_dataset(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        ctl_path=args.ctl_path,
        split=args.split,
        years=list(range(args.start_year, args.end_year + 1)),
        list_vars=VARS,
    )


if __name__ == "__main__":
    main()