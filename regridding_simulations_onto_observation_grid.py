# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:32:24 2019

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


simulation_dir = 'C://Users/mp877190/Desktop/PAPERS/2nd_paper_ERL/simulation_files'
landsat_dir = 'C://Users/mp877190/Desktop/PAPERS/2nd_paper_ERL/landsat_files'


def read_in_ncfile(dir_name, file_name):
    nc_name = os.path.join(dir_name, file_name)
    dataset = nc.Dataset(nc_name)
    lat = np.array(dataset.variables['latitude'])
    lon = np.array(dataset.variables['longitude'])
    print("lat, shape", lat.shape)
    print("lon shape", lon.shape)
    return dataset, lat, lon

def distance(x1,y1, x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  


def regrid_sim_onto_sat(sat_lat, sat_lon, sim_lat, sim_lon, sim_data):
    #output_array = np.zeros((sat_lat.shape[0]-1, sat_lat.shape[1]-1)) #np.zeros_like(sat_lat)
    output_array = np.zeros((sat_lat.shape[0], sat_lat.shape[1])) #np.zeros_like(sat_lat)
    #lat_nan_condition = (sat_lat[:,0] > np.amax(sim_lat[:,0]))*(sat_lat[:,0] < np.amin(sim_lat[:,0]))
    #lon_nan_condition = (sat_lon[0,:] > np.amax(sim_lon[0,:]))*(sat_lon[0,:] < np.amin(sim_lon[0,:]))
    #output_array = np.where(~lat_nan_condition, sim_data[], np.nan)
    
    for i in range(sat_lat[:-1,0].size): #looking only at the bottom and the left edge of the large grid
        for j in range(sat_lon[0,:-1].size): #looking only at the bottom and the left edge of the large grid
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                big_grid_condition = (sim_lat <= sat_lat[i,0])*(sim_lat >= sat_lat[i+1,0])*(sim_lon >= sat_lon[0,j])*(sim_lon <= sat_lon[0,j+1])
                output_array[i,j] = np.nanmean(np.where(big_grid_condition, sim_data, np.nan))
                #print(output_array[i,j])
    print("regridded_data", output_array.shape)
    return output_array


def create_netcdf(output_name, regridded_data_2d, regridded_lat_array, regridded_lon_array):
	#open a new netcdf file in the write ('w') mode
	dataset = nc.Dataset(output_name+'.nc', 'w', format='NETCDF4_CLASSIC')

	#create dimensions
	lat = dataset.createDimension('lat', len(regridded_lat_array[:,0])) 
	lon = dataset.createDimension('lon', len(regridded_lon_array[0,:])) 
	time = dataset.createDimension('time', None) #unlimited time, if we want to add data later
	
	time = dataset.createVariable('time', np.float32, ('time',))
	time.standard_name = 'time'
	#bands = dataset.createVariable('bands', np.int32, ('band',))
	latitudes = dataset.createVariable('latitude', np.float32, ('lat','lon',))
	latitudes.standard_name = 'latitude'
	longitudes = dataset.createVariable('longitude', np.float32, ('lat','lon',))
	longitudes.standard_name = 'longitude'

	# Create the actual 4-d variable
	temperature = dataset.createVariable('temperature', np.float32, ('lat','lon',))

	# Variable Attributes
	latitudes.units = 'degree_north'
	longitudes.units = 'degree_east'
	#bands.units = 'micro_meters'
	temperature.units = 'deg C'
	
	#Assign values to the variables
	lats = regridded_lat_array #np.flipud(regridded_data_3d[0,:,:]) #np.arange(-90,91,2.5)
	lons = regridded_lon_array #regridded_data_3d[1,:,:]  #np.arange(-180,180,2.5)
	latitudes[:,:] = lats[:,:]
	longitudes[:,:] = lons[:,:]
	print(latitudes)
	print(longitudes)

	temperature[:,:] = regridded_data_2d[:,:] #regridded_data_3d[2,:,:]

	print(temperature)
	dataset.close()
	return

################################################################################

dataset = nc.Dataset('C:/Users/mp877190/Desktop/PAPERS/2nd_paper_ERL/simulation_files/heysham_flood_south_actual_flow.nc')
print(dataset.dimensions.keys())
print(dataset.variables.keys())
print(dataset.variables['temperature'])
temp_1 = np.array(dataset.variables['temperature'][:,:])
plt.imshow(temp_1)
plt.colorbar()
plt.clim(20,30)
plt.show()
"""
landsat_file, landsat_lat, landsat_lon = read_in_ncfile(landsat_dir, '20171201_flood_south.nc')
simulation_file, simulation_lat, simulation_lon = read_in_ncfile(simulation_dir, 'heysham_flood_south_actual_flow.nc')

print(landsat_lat) #(np.amin(landsat_lat), np.amax(landsat_lat))
print(simulation_lat) #(np.amin(simulation_lat), np.amax(simulation_lat))

print(landsat_lon)
print(simulation_lon)

simulation_sst = np.array(simulation_file.variables["temperature"])
landsat_bt = np.array(landsat_file.variables["BT_band10"])

t = time.process_time()
sat_sim_sst = regrid_sim_onto_sat(landsat_lat, landsat_lon, simulation_lat, simulation_lon, simulation_sst[1,:,:])
elapsed_time = time.process_time() - t
print(elapsed_time)

print(simulation_sst[1,:,:])
print(sat_sim_sst)

create_netcdf('C://Users/mp877190/Desktop/20171201_flood_south', sat_sim_sst, landsat_lat, landsat_lon)

plt.imshow(simulation_sst[1,:,:])
plt.colorbar()
plt.clim(20, 25)
plt.show()

plt.imshow(sat_sim_sst[100:250,100:250])
plt.colorbar()
plt.clim(20,25)
plt.show()

plt.imshow(np.rot90(landsat_bt[100:250, 100:250]))
plt.colorbar()
plt.clim(290,295)
plt.show()
"""

