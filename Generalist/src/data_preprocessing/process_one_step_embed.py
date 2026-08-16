import os
from tqdm import tqdm
import numpy as np
import argparse

target_lon, target_lat = np.arange(97.06, 122.39, 0.03), np.arange(17.2, 30.41, 0.03)

def get_localisation_embeddings(lon, lat):
    """Get localisation embeddings.

    Parameters
    ----------
    mesh : torch.Tensor
        Latitude  \in [-2/pi, 2/pi]
        Lontitude \in [-pi, pi] 
        (2, 32, 64)

    Returns
    -------
    torch.Tensor (3, 441, 845)
    """
    # Converting to radians
    cos_lat = np.cos(lat)
    cos_lon, sin_lon = np.cos(lon), np.sin(lon)

    return np.stack(
        [
            cos_lat,
            cos_lon,
            sin_lon,
        ],
        axis=0
    )

def get_time_embeddings(day_of_year, hour_of_day):
    """Returns the day and season embeddings.

    Parameters
    ----------
    day_of_year_ratio : torch.Tensor
        The day of the year divided by the number of days in the specified year.
    hour_of_day : torch.Tensor
        The hour of the day.

    Returns
    -------
    torch.Tensor: (nb_time_step, 4)
    """
    cos_doy, sin_doy = np.cos(day_of_year), np.sin(day_of_year)
    cos_hod, sin_hod = np.cos(hour_of_day), np.sin(hour_of_day)
    return np.stack(
        [
            cos_doy,
            sin_doy,
            cos_hod,
            sin_hod,
        ],
        axis=1
    )

def get_time_localisation_embeddings(
    day_of_year, hour_of_day, lon, lat, terrain, dem
):
    """Get join time-localisation embeddings for every $t$ in the time step range.
    Embedding are in the same order as in the paper:
    * Day and season $\psi(t)$: (4, T)
    * Localisation $\psi(x)$: (6, H, W)
    * lat, lon: (H, W)
    * Land sea mask: (H, W)
    * Terrain: (H, W)

    Parameters
    ----------
    day_of_year_ratio : torch.Tensor
        The day of the year divided by the number of days in the specified year.
    hour_of_day : torch.Tensor
        The hour of the day.
    lat : torch.Tensor
        latitude in 1D (32)
    lon : torch.Tensor
        longitude in 1D (64)
    lsm : torch.Tensor
        Land sea mask constant (1, 32, 64)
    oro : torch.Tensor
        Orography constant (1, 32, 64)

    Returns
    -------
    torch.Tensor
        (nb_time_step, 38, 32, 64)
    """

    nb_time_step = len(hour_of_day)
    loc_emb = get_localisation_embeddings(lon=lon, lat=lat)[None, :, :, :]  # (1, 3, 441, 845)
    day_seas_emb = get_time_embeddings(
        2 * np.pi * day_of_year, 2 * np.pi * hour_of_day
    )[:,:, None,None]  # (nb_time_step，4, 1, 1)

    # Preparing for localization and time embeddings combination
    loc_emb = loc_emb.repeat(nb_time_step, axis=0) # (1, 3, 441, 845) -> (nb_time_step, 3, 441, 845)
    day_seas_emb = day_seas_emb.repeat(441, axis=2).repeat(845, axis=3) # day_seas_emb: (nb_time_step, 4) -> (nb_time_step, 4, 441, 845)

    return np.concatenate(
        [
            day_seas_emb,  # (nb_time_step, 4, 441, 845)
            loc_emb,  # (nb_time_step, 3, 441, 845)
            terrain[None, None, :, :].repeat(nb_time_step, axis=0),  # (441, 845) -> (nb_time_step, 1, 441, 845)
            dem[None, None, :, :].repeat(nb_time_step, axis=0), # (441, 845) -> (nb_time_step, 1, 441, 845)
        ],
        axis=1,
    )

def create_one_step_embedding(root_dir, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    hour_of_year = np.arange(0, 8784)
    day_of_year = hour_of_year // 24
    hour_of_day = hour_of_year % 24

    day_of_year = day_of_year / 366
    hour_of_day = hour_of_day / 24

    lon, lat = np.meshgrid(target_lon, target_lat)
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    terrain = np.load(os.path.join(root_dir, 'terrain.npy'))
    dem = np.load(os.path.join(root_dir, 'dem.npy'))
    # lsm = np.load(os.path.join(root_dir, 'lsm.npy'))

    time_pos_embedding = get_time_localisation_embeddings(
                            day_of_year, 
                            hour_of_day, 
                            lon, 
                            lat, 
                            terrain, 
                            dem, 
    )  # .to(device=device)  # if enough VRAM

    # with tqdm(total=(len(time_pos_embedding))) as pbar:
    for idx in tqdm(range(len(time_pos_embedding)), desc='hour', position=0):
        save_path = os.path.join(save_dir, f'{idx:04}.npy')
        np.save(save_path, time_pos_embedding[idx]) # Avoid shared memory


def parse_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save regridded files.')
    
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    create_one_step_embedding(root_dir=args.root_dir,
                              save_dir=args.save_dir)