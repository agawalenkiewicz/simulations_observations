# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:48:03 2019

@author: mp877190
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import re
import pandas as pd
from datetime import datetime
from scipy import interpolate
import scipy

from itertools import islice 

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import cv2
import netCDF4 as nc



def get_contour(img, width = 1):
    for i in range(width):
        img = scipy.misc.imfilter(img, "find_edges")/255
    return img



def create_netcdf(output_name, regridded_data_3d, regridded_lat_array, regridded_lon_array):
	#open a new netcdf file in the write ('w') mode
	dataset = nc.Dataset(output_name+'.nc', 'w', format='NETCDF4_CLASSIC')

	#create dimensions
	lat = dataset.createDimension('lat', len(regridded_lat_array[:,0])) 
	lon = dataset.createDimension('lon', len(regridded_lon_array[0,:])) 
	depth = dataset.createDimension('depth', len(regridded_data_3d[:]))
	time = dataset.createDimension('time', None) #unlimited time, if we want to add data later
	
	time = dataset.createVariable('time', np.float32, ('time',))
	time.standard_name = 'time'
	#bands = dataset.createVariable('bands', np.int32, ('band',))
	latitudes = dataset.createVariable('latitude', np.float32, ('lat','lon',))
	latitudes.standard_name = 'latitude'
	longitudes = dataset.createVariable('longitude', np.float32, ('lat','lon',))
	longitudes.standard_name = 'longitude'

	# Create the actual 4-d variable
	temperature = dataset.createVariable('temperature', np.float32, ('depth', 'lat','lon'))

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
	for i in range(len(regridded_data_3d)) :
		temperature[i,:,:] = regridded_data_3d[i] #regridded_data_3d[2,:,:]

	print(temperature)
	dataset.close()
	return
#################################################     
flood_dir = 'C://Users/mp877190/Desktop/28_10_2019/heysham_flood_north_actual_conditions_2_txt_files'

flood_s_dir = 'C://Users/mp877190/Desktop/28_10_2019/heysham_flood_south_max_conditions_txt_files'

ebb_dir = 'C://Users/mp877190/Desktop/28_10_2019/heysham_ebb_long_actual_conditions_2_txt_files'

flood_depths = ['x_all_y_all_z_3_3.txt', 'x_all_y_all_z_1_9.txt', 'x_all_y_all_z_0_5.txt', 'x_all_y_all_z_-0_9.txt', 'x_all_y_all_z_-2_3.txt', 'x_all_y_all_z_-3_7.txt', 'x_all_y_all_z_-5_1.txt', 'x_all_y_all_z_-6_5.txt', 'x_all_y_all_z_-7_9.txt', 'x_all_y_all_z_-9_3.txt']

ebb_depths = ['x_all_y_all_z_1.txt', 'x_all_y_all_z_-1.txt', 'x_all_y_all_z_-3.txt', 'x_all_y_all_z_-5.txt', 'x_all_y_all_z_-7.txt', 'x_all_y_all_z_-9.txt']


files = [] 
depths = ebb_depths
dir_name = ebb_dir

for i in depths:
    file = os.path.join(dir_name, i)
    files.append(file)


x_location = []
y_location = []
depth = []
temperature = []

for j,file in enumerate(files):
    #textfile = open(file, 'r')
    #all_lines = textfile.readlines()
    #print(len(all_lines)) 
    x_location.append([])
    y_location.append([])
    depth.append([])
    temperature.append([])  
    
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i > 8 and i < 276009: #276009 for ebb #320009 for flood
                #print(line)
                line = line.strip('\n')
                #content = line.replace(' ','\t')
                #words = content.split('\t')
                words = line.split()
                words = [np.nan if word == 'NAN' else word for word in words]
                x_location[j].append(np.float(words[0]))
                y_location[j].append(np.float(words[1]))
                depth[j].append(np.float(words[2]))
                temperature[j].append(np.float(words[3]))
                
        x_location[j] = np.array(x_location[j]).reshape(460,600) #460,600 for ebb #500,640 for flood
        y_location[j] = np.array(y_location[j]).reshape(460,600)
        temperature[j] = np.flipud(np.array(temperature[j]).reshape(460,600))
        depth[j] = np.array(depth[j]).reshape(460,600)


temperature_2d_field = np.array(temperature[1]) #.reshape(460,600)
image_thresholded = np.copy(temperature_2d_field)
image_thresholded[image_thresholded<20.7] = np.nan
where_are_NaNs = np.isnan(image_thresholded)
image_thresholded[where_are_NaNs] = 0
image_thresholded[image_thresholded>0] = 1.0

#idx_for_land = list(zip(*np.where(temperature_2d_field[110:130,235:245] == np.amin(temperature_2d_field))))

temp_field_masked = np.ma.masked_where(temperature_2d_field <= 20.5, temperature_2d_field)
#depth_for_temp_field = np.ma.masked_where(temperature_2d_field <= 20.5, np.array(depth[1]))

image_contour = get_contour(image_thresholded)

plt.imshow(temperature_2d_field) #image_thresholded))
#plt.imshow(np.flipud(image_contour))
#plt.contour(x_location[1], y_location[1], image_thresholded, levels=[0.5])
#plt.plot([248],[73], 'r.') #([243],[373], 'r.') #([436],[189], 'r.')
plt.colorbar()
plt.clim(20,25)
plt.show()

flood_lats = np.linspace(54.050416, 54.025466, 500) 
flood_lons= np.linspace(-2.936012, -2.904062, 640) 

ebb_lats = np.linspace(54.041216, 54.018266, 460)
ebb_lons = np.linspace(-2.945662, -2.915712, 600)

flood_s_lats = np.linspace(54.035416, 54.010466, 500)
flood_s_lons = np.linspace(-2.936262, -2.904312, 640)

print(flood_lats[373], (flood_lats[3] - flood_lats[2]))
print(flood_lons[243], (flood_lons[3] - flood_lons[2]))

print(ebb_lats[189], (ebb_lats[3] - ebb_lats[2]))
print(ebb_lons[436], (ebb_lons[3] - ebb_lons[2]))

print(flood_s_lats[73], (flood_s_lats[3] - flood_s_lats[2]))
print(flood_s_lons[248], (flood_s_lons[3] - flood_s_lons[2]))



lons_2d, lats_2d = np.meshgrid(ebb_lons, ebb_lats) #np.meshgrid(flood_lons, flood_lats)



nc_name = os.path.join(dir_name, 'heysham_ebb_long_actual_flow')
create_netcdf(nc_name, temperature, lats_2d, lons_2d)

"""
dataset = nc.Dataset(nc_name)
print(dataset.dimensions.keys())
print(dataset.variables.keys())
print(dataset.variables['temperature'])
temp_1 = np.array(dataset.variables['temperature'][1,:,:])
plt.imshow(temp_1)
plt.colorbar()
plt.clim(20,30)
plt.show()
"""
###################################################################
"""
#Plot in 3-D
fig = plt.figure()
ax = fig.gca(projection='3d')

print(np.array(x_location[0][0,:]))
print(np.array(y_location[0][:,0]))


# Make data.
X = np.array(x_location[0][0,:]) #np.arange(np.array(x_location[0][0,:]))
Y = np.array(y_location[0][:,0]) #np.arange(np.array(y_location[0][:,0]))

X, Y = np.meshgrid(X, Y)
temp_at_depth = np.sqrt(X**2 + Y**2)
Z = temperature[1] #np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-9.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
"""