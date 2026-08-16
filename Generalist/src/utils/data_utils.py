CONSTANTS = [
    'zs',
]

SINGLE_LEVEL_VARS = [
    't2m',
    'ts',
    'q2m',
    'psl',
    'u10m',
    'v10m',
    'ps',
]

PRESSURE_LEVEL_VARS = [
    'u',
    'v',
    't',
    'h',
    'Qv',
]

DEFAULT_PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

NAME_TO_VAR = {
    "mean_surface_latent_heat_flux": "mslhf",
    "mean_surface_net_long_wave_radiation_flux": "msnlwrf",
    "mean_surface_net_short_wave_radiation_flux": "msnswrf",
    "mean_surface_sensible_heat_flux": "msshf",
    "mean_top_downward_short_wave_radiation_flux": "mtdnswrf",
    "mean_top_net_long_wave_radiation_flux": "mtnlwrf",
    "mean_top_net_short_wave_radiation_flux": "mtnswrf",
    "skin_temperature": "skt",
    "snow_depth": "snd",
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "10m_wind_speed": "w10",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr",
    "toa_incident_solar_radiation_6hr": "tisr_6hr",
    "toa_incident_solar_radiation_12hr": "tisr_12hr",
    "toa_incident_solar_radiation_24hr": "tisr_24hr",
    "total_precipitation": "tp",
    "total_precipitation_6hr": "tp_6hr",
    "total_precipitation_12hr": "tp_12hr",
    "total_precipitation_24hr": "tp_24hr",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "slt": "slt",
    "lattitude": "lat2d",
    "longitude": "lon2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vertical_velocity": "vel",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
    "vorticity": "vo",
    "potential_vorticity": "pv",
    "total_cloud_cover": "tcc",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

NAME_LEVEL_TO_VAR_LEVEL = {}

for var in SINGLE_LEVEL_VARS:
    if var in NAME_TO_VAR:
        NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

for var in PRESSURE_LEVEL_VARS:
    if var in NAME_TO_VAR:
        for l in DEFAULT_PRESSURE_LEVELS:
            NAME_LEVEL_TO_VAR_LEVEL[var + "_" + str(l)] = NAME_TO_VAR[var] + "_" + str(l)

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}

VARIABLES=[
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

# Pressure-level weights for training Stormer
single_level_weight_dict = {
    'cmpa': 0.1,
    't2m': 1.0,
    'ts': 1.0,
    'q2m': 1.0,
    'psl': 1.0,
    'u10m': 0.1,
    'v10m': 0.1,
    'ps': 0.1,
}

pressure_weights = [l / sum(DEFAULT_PRESSURE_LEVELS) for l in DEFAULT_PRESSURE_LEVELS]
pressure_level_weight_dict = {}
for var in PRESSURE_LEVEL_VARS:
    for l, w in zip(DEFAULT_PRESSURE_LEVELS, pressure_weights):
        pressure_level_weight_dict[var + "_" + str(l)] = w

# pressure_level_weight_dict['u_1000'] = 0.1
# pressure_level_weight_dict['v_1000'] = 0.1

WEIGHT_DICT = {**single_level_weight_dict, **pressure_level_weight_dict}