# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:00:57 2019

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

def centered_average(nums):
	return (np.nansum(nums) - np.nanmax(nums) - np.nanmin(nums)) / (np.count_nonzero(~np.isnan(nums)) - 2)

def morphological_dilation(masked_image, n): #n=3
    """
    Extending the landmask.
    Should extend the landmask over the image for 3 pixels (0.0015 degrees)
    ----------
    from stackoverflow:
    def dilation(a, n):
    m = np.isnan(a)
    s = np.full(n, True, bool)
    return ndimage.binary_dilation(m, structure=s, origin=-(n//2))
    -----------
    For sparse initial masks and small n this one is also pretty fast:
    def index_expansion(a, n):
    mask = np.isnan(a)
    idx = np.flatnonzero(mask)
    expanded_idx = idx[:,None] + np.arange(1, n)
    np.put(mask, expanded_idx, True, 'clip')
    return mask
    """
    mask = np.isnan(masked_image)
    s = ndimage.morphology.generate_binary_structure(2, 1)
    extended_mask = ndimage.binary_dilation(mask, structure=s, iterations=3).astype(mask.dtype)
    return extended_mask

def choose_plume(image_thresholded):	
    #now find the objects
    where_are_NaNs = np.isnan(image_thresholded)
    image_thresholded[where_are_NaNs] = 0
	
    labeled_image, numobjects = ndimage.label(image_thresholded)
	
    plt.imshow(labeled_image)
    plt.show()
    object_areas = np.bincount(labeled_image.ravel())[:]
    #to exclude the first object which is background , index from 1
    object_idx = [i for i in range(1, numobjects) if object_areas[i] > 3]
    print('object area' , object_areas)
    print('object idx' , object_idx)
    # Remove small white regions
    #labeled_image = ndimage.binary_opening(labeled_image)
    # Remove small black hole
    #labeled_image = ndimage.binary_closing(labeled_image)
    chosen_object = [0,50]
    for object in object_idx: #range(0,numobjects):
        #object = object + 1
        #print('object' , object)
        iy, ix = np.where(labeled_image == object)
        centridx_y = 200
        centridx_x = 200
        min_dist = np.min(np.sqrt((np.abs(centridx_y - iy))**2 + (np.abs(centridx_x - ix))**2))
        if min_dist < chosen_object[1]:
            chosen_object = [object, min_dist]
        #print(object , min_dist)
    #print('Chosen object' , chosen_object)
    #chosen_plume = np.where((labeled_image == chosen_object[0]), chosen_object[0], 0)
    if chosen_object[0] == 50:
        chosen_plume = np.zeros_like(labeled_image)
    else:
        chosen_plume = np.where((labeled_image == chosen_object[0]), 1., 0.)
        #chosen_plume = np.where((labeled_image == chosen_object[0]), 0., 1.)
    area = sum(sum(i == True for i in chosen_plume))
    area_km = np.float(area) * 0.001033
    print("Detected plume area (number of pixels):" , area , area_km)
    return chosen_plume


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
##############################################################################

observation_file = 'C://Users/mp877190/Desktop/files_for_comparison_same_grid_and_temp/obs_files_after_oe/heysham_flood_20171201.nc'
simulation_file = 'C://Users/mp877190/Desktop/files_for_comparison_same_grid_and_temp/simulation_files_on_the_obs_grid/20171201_flood_south.nc'

sim_sst = read_in_nc_file(simulation_file)
obs_sst = np.flipud(read_in_nc_file(observation_file))


obs_sst[obs_sst<0.] = np.nan       
ambient = centered_average(obs_sst[200:250, 0:100])
print("ambient", ambient)
threshold = np.float(ambient) + 1.5
image_thresh = np.copy(obs_sst)
image_thresh[image_thresh<threshold] = np.nan
#extended_landmask = morphological_dilation(image_thresh, 20)
#image_thresh[extended_landmask] = np.nan
obs_plume = choose_plume(image_thresh)

plt.imshow(sim_sst)
plt.colorbar()
plt.clim(0,10)
plt.show()

image_thresh_2 = np.copy(sim_sst)
threshold_2 = 20.2
image_thresh_2[image_thresh_2 < threshold_2] = np.nan
image_thresh_2[np.isnan(image_thresh_2)] = 0.
image_thresh_2[image_thresh_2 > 0.] = 2.
sim_plume = np.roll(image_thresh_2, 0, axis=1) # left
sim_plume = np.roll(sim_plume, 30, axis=0) # down
print(sim_plume.shape)
print(obs_plume.shape)


#added_plumes = [map(sum, zip(*t)) for t in zip(sim_plume, obs_plume)] 
added_plumes = [[obs_plume[i][j] + sim_plume[i][j]  for j in range(len(obs_plume[0]))] for i in range(len(obs_plume))] 

plt.imshow(np.flipud(added_plumes))
plt.colorbar()
plt.clim()
plt.xlim(150,220)
plt.ylim(150,220)
plt.title("Thermal plume during flood tide - 01/12/2017 \n")
plt.show()

totals_obs = Counter(i for i in list(itertools.chain.from_iterable(obs_plume)))
totals_sim = Counter(i for i in list(itertools.chain.from_iterable(sim_plume)))
totals = Counter(i for i in list(itertools.chain.from_iterable(added_plumes)))
print("PRINTING STATS FOR THE COMPARISON")
print("Plume observed by the Landsat 8 satellite: %d pixels" % (totals_obs[1.]))
print("Plume simulated by the FLOW-3D CFD software: %d pixels" % (totals_sim[2.]))
print("Number of pixels marked by observations and simulations as plume: %d pixels" % (totals[3.]))
print("Number of observed plume pixels, not present is simulations: %d pixels" % (totals[1.]))
print("Number of simulated plume pixels, not present in observations: %d pixels" % (totals[2.]))
print("\n")
print("Hit rate: %f " % (totals[3.]/totals_obs[1.]))
print("False alarm rate: %f" % ((totals_sim[2.]-totals[3.])/totals_sim[2.]))
print("Misses: %f " % ((totals_obs[1.]-totals[3.])/totals_obs[1.]))