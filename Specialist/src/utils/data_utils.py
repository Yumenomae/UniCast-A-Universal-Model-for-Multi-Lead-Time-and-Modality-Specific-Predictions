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

WEIGHT_DICT = {**single_level_weight_dict, **pressure_level_weight_dict}