# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:13:46 2020

@author: mp877190
"""

import numpy as np
import pylab
import netCDF4 as nc
import matplotlib.pyplot as plt
import math
from statistics import mean
import sys, os
import warnings
import time
from scipy import ndimage

from collections import Counter
import itertools



def read_in_nc_file(name):
    dataset = nc.Dataset(name)
    #print(dataset.dimensions.keys())
    #print(dataset.variables.keys())
    #print(dataset.variables['temperature'])
    temp = np.array(dataset.variables['temperature'][:,:])
    return temp

def temperature_into_density(temperature_2d_field):
    # create empty array of the same size as 2-D temperature field
    density_2d_field = np.copy(temperature_2d_field)
    # equation from McCutcheon et al. 1993 (water density as function of temperature and concentration)
    for T in np.nditer(density_2d_field, op_flags=['readwrite']):
        T[...] = 1000*(1 - (T+288.9414)/(508929.2*(T+68.12963))*(T-3.9863)**2)
        
    return density_2d_field

def density_delta(density_2d_field):
    density_at_20_degrees = 998.234
    for density in np.nditer(density_2d_field, op_flags=['readwrite']):
        density[...] = density - density_at_20_degrees
    return density_2d_field

################################################################################
################################################################################
simulation_file = 'C://Users/mp877190/Desktop/files_for_comparison_same_grid_and_temp/simulation_files_on_the_obs_grid/20171201_flood_south.nc'
sim_sst = read_in_nc_file(simulation_file)

plt.imshow(sim_sst)
plt.xlim(150,220)
plt.ylim(150,220)
plt.colorbar()
plt.clim(20,25)
plt.show()


density = temperature_into_density(sim_sst)
print(density)
print(np.nanmin(density), np.nanmax(density))

density_diff = density_delta(density)


plt.imshow(density_diff)
plt.xlim(150,220)
plt.ylim(150,220)
plt.colorbar()
plt.clim(-1,1)
plt.show()


"""
an_array = np.arange(400).reshape(20,20)
print(an_array)
for item in np.nditer(an_array, op_flags=['readwrite']):
    item[...] = 1000*(1 - (item+288.9414)/(508929.2*(item+68.12963))*(item-3.9863)**2)
print(an_array)
"""
        