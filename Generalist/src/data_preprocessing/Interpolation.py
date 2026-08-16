import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


grapes_lon = np.arange(93.04, 132.95, 0.03)
grapes_lat = np.arange(10, 43.49, 0.03)

intercept_grapes_lon = np.arange(97.06, 122.39, 0.03)
intercept_grapes_lat = np.arange(17.2, 30.41, 0.03)
intercept_grapes_lat[-1] = 30.4

cmpa_lon = np.arange(97.05, 122.41, 0.05)
cmpa_lat = np.arange(17.2, 30.41, 0.05)
cmpa_lat[-1] = 30.4

grapes_lon, grapes_lat = np.meshgrid(grapes_lon, grapes_lat)
intercept_grapes_lon, intercept_grapes_lat = np.meshgrid(intercept_grapes_lon, intercept_grapes_lat)
cmpa_lon, cmpa_lat = np.meshgrid(cmpa_lon, cmpa_lat)

def regrid_GRAPES(data):
    for var, values in data['input'].items():
        values_new = griddata((grapes_lon.flatten(), grapes_lat.flatten()), values.flatten(), (cmpa_lon, cmpa_lat), method='linear')
        data['input'][var] = values_new
    return data

def regrid_CMPA(data):
    data = data[::-1, :]
    regrid = griddata((cmpa_lon.flatten(), cmpa_lat.flatten()), data.flatten(), (intercept_grapes_lon, intercept_grapes_lat), method='linear')
    return regrid
