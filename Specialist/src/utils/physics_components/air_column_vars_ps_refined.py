import torch
import numpy as np
from typing import Any, Dict, Tuple, Dict, List, Sequence
from scipy.interpolate import griddata
import h5py
from .plot import visualization, visualization_vars

def get_data_given_path(
    path: str
) -> torch.tensor:
    """Prepare the data given the data path and variables.
    
    """
    with h5py.File(path, 'r') as f:
        data = {
            main_key: {
                sub_key: np.array(value) for sub_key, value in group.items()
        } for main_key, group in f.items() if main_key in ['input']}

    x = [data['input'][v] for v in variables]
    return np.stack(x, axis=0)

def get_data_given_path_(
    path: str
) -> torch.tensor:
    """Prepare the data given the data path and variables.
    
    """
    with h5py.File(path, 'r') as f:
        data = {
            main_key: {
                sub_key: np.array(value) for sub_key, value in group.items()
        } for main_key, group in f.items() if main_key in ['input']}

    return data['input']

class compute_air_column_variable_taking_geography():
    """Compute the derived air-column variables and its physical effect.

    Implements 10 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def integration_of_pressure(self):
    # Integration along air pressure

    def pressure_level_reshape(self):
    # Decomposition of inputs into surface variables as well as pressure-level variables

    def air_column_humidity_temperature(self):
    # Calculate the humidity-temperature of in the air column at current time t0

    def air_column_precipitation(self):
    # Calculate the amount of precipitable water in the air column at current time t0

    def compute_advection(self):
    # Calculate the advection term.

    def air_column_humidity_temperature_advection_integration(self):
    # Calculate the integration of advection term of air-column humidity-temperature in pressure level.

    def air_column_precipitation_advection_integration(self):
    # Calculate the integration of advection term of air-column temperature in pressure level.

    def predict_air_column_humidity_temperature(self):
    # Forecasting air column level humidity-temperature based on advection equation.

    def predict_air_column_precipitation(self):
    # Forecasting air-column precipitation based on advection equation.    

    def predict_precipitation(self):
    # Forecasting precipitation based on air-column precipitation and advection equation. 
    ```

    """
    def __init__(
        self,
        pressure_level: List,
        lat: np.array,
        lon: np.array,
        a: int,
    ) -> None:
        """Initialize the integration variables.
        """
        super().__init__()
        self.pres = torch.tensor(pressure_level) * 100 # Pa
        lon, lat = np.meshgrid(lon, lat)
        lon, lat = np.deg2rad(lon), np.deg2rad(lat) # rad
        self.lon = torch.tensor(lon)
        self.lat = torch.tensor(lat)

        self.r = 6731000 # m
        self.L = None
        self.Cp = None
        self.Rd = 287.05 # J/(kg·K)
        self.p0 = None
        self.g = 9.80665 # (m/s²‌)

        self.a = a

    def integration_of_pressure_taking_geography(self, x: torch.Tensor, surface_x: torch.Tensor, ps: torch.Tensor):
        # x: (B, 13, 441, 845), surface_x: (B, 441, 845)
        x = torch.concat([surface_x.unsqueeze(1), x], dim=1) # (B, 14, 441, 845)
        p = self.pres.view(1, 13, 1, 1).expand(x.shape[0], 13, x.shape[2], x.shape[3]).to(x.device) # (B, 13, 441, 845)
        p = torch.concat([ps.unsqueeze(1) * 100, p], dim=1)   # (B, 14, 441, 845)
        sorted_p, sorted_indices = p.sort(dim=1, descending=True)
        x = x.gather(1, sorted_indices)

        ranks = torch.zeros_like(sorted_indices, dtype=torch.long)
        ranks.scatter_(dim=1, index=sorted_indices, src=torch.arange(14).view(1, 14, 1, 1).expand(x.shape[0], 14, x.shape[2], x.shape[3]).to(x.device))
        p0_rank = ranks[:, 0, :, :]

        mean_x = (x[:,:13] + x[:, 1:]) / 2
        delta_p = sorted_p[:, :13] - sorted_p[:, 1:] 

        indices = torch.arange(13).view(1, 13, 1, 1).expand(x.shape[0], 13, x.shape[2], x.shape[3]).to(x.device)
        Mask = (indices >= p0_rank.unsqueeze(1)).float()

        output = (mean_x * delta_p * Mask).sum(dim=1)  # (B, 441, 845)
        return output
    
    def pressure_level_reshape_denorm(self, x: torch.Tensor, reverse_inp_transform) -> Dict:
        if reverse_inp_transform:
            x = reverse_inp_transform(x)
        surface = x[:, :8, :, :]
        upper_air = x[:, 8:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])   # (B, 5, 13, 441, 845)

        upper_vars = ['u', 'v', 't', 'h', 'Qv']
        surface_vars = ['cmpa','t2m','ts','q2m','psl','u10m','v10m','ps']
        upper_vars_dict, surface_vars_dict = {}, {}
        for idx, var in enumerate(upper_vars):
            upper_vars_dict[var] = upper_air[:, idx]
        
        for idx, var in enumerate(surface_vars):
            surface_vars_dict[var] = surface[:, idx]

        return upper_vars_dict, surface_vars_dict

    def air_column_humidity(self, Qv: torch.Tensor, q2m: torch.Tensor, ps: torch.Tensor):
        int_press_humidity = self.integration_of_pressure_taking_geography(Qv, q2m, ps)   # (B, 441, 845)
        return int_press_humidity
    
    def compute_advection(self, x: torch.Tensor, surface_x: torch.Tensor, 
                          u: torch.Tensor, v: torch.Tensor, surface_u: torch.Tensor, surface_v: torch.Tensor,
                          t: torch.Tensor, surface_t: torch.Tensor,
                          ps: torch.Tensor):
        lat = self.lat[None, None, :, :].to(x.device)     # (1, 1, 441, 845)

        pres = self.pres.view(1, 13, 1, 1).expand(x.shape[0], 13, x.shape[2], x.shape[3]).to(x.device)
        atmosDen = (pres / t) * (1 / self.Rd) # (B, 13, H, W)
        surface_atmosDen = (ps / surface_t) * (1 / self.Rd) # (B, H, W)
        rxu, rxv = atmosDen * x * u, atmosDen * x * v
        surface_rxu, surface_rxv = surface_atmosDen * surface_x * surface_u, surface_atmosDen * surface_x * surface_v

        partial_rxu_rectified_lon = torch.gradient(rxu, spacing = 0.03 * torch.pi / 180, dim=3)[0] / (self.r * torch.cos(lat))
        partial_rxv_rectified_lat = torch.gradient(rxv, spacing = 0.03 * torch.pi / 180, dim=2)[0] / self.r
        div = (partial_rxu_rectified_lon + partial_rxv_rectified_lat) * (1 / atmosDen)

        partial_surface_rxu_rectified_lon = torch.gradient(surface_rxu, spacing = 0.03 * torch.pi / 180, dim=2)[0] / (self.r * torch.cos(lat)[:, 0])
        partial_surface_rxv_rectified_lat = torch.gradient(surface_rxv, spacing = 0.03 * torch.pi / 180, dim=1)[0] / self.r
        surface_div = (partial_surface_rxu_rectified_lon + partial_surface_rxv_rectified_lat) * (1 / surface_atmosDen)

        return div, surface_div
    

    def air_column_humidity_advection_integration(self, 
                                                  Qv: torch.Tensor, q2m: torch.Tensor,
                                                  u: torch.Tensor, v: torch.Tensor, u10m: torch.Tensor, v10m: torch.Tensor,
                                                  t: torch.Tensor, t2m: torch.Tensor,
                                                  ps: torch.Tensor):
        
        div, surface_div = self.compute_advection(Qv, q2m, u, v, u10m, v10m, t, t2m, ps)
        int_press_div = self.integration_of_pressure_taking_geography(div, surface_div, ps)   # (B, 441, 845)

        return - int_press_div



    def predict_air_column_humidity(self,
                                        Qv: torch.Tensor, 
                                        u: torch.Tensor, v: torch.Tensor,
                                        u10m: torch.Tensor, v10m: torch.Tensor,
                                        q2m: torch.Tensor, ps: torch.Tensor,
                                        t: torch.Tensor, t2m: torch.Tensor,
                                        fute_precp: torch.Tensor):
        ac_humidity = self.air_column_humidity(Qv, q2m, ps)
        partial_t_ac_humidity = self.air_column_humidity_advection_integration(Qv, q2m, u, v, u10m, v10m, t, t2m, ps)
        pred = ac_humidity + self.a * partial_t_ac_humidity * 3600 - fute_precp * self.g
        return pred
    