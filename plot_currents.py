# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:26:32 2019

@author: mp877190
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

file1 = 'C:/Users/mp877190/Desktop/MET_Norway_current_data/11am/roms_nordic4_ZDEPTHS_hr.an.20180125.nc'
file2 = 'C:/Users/mp877190/Desktop/MET_Norway_current_data/11am/roms_nordic4_ZDEPTHS_hr.an.20171208.nc'
file3 = 'C:/Users/mp877190/Desktop/MET_Norway_current_data/11am/roms_nordic4_ZDEPTHS_hr.an.20171201.nc'
file4 = 'C:/Users/mp877190/Desktop/MET_Norway_current_data/11am/roms_nordic4_ZDEPTHS_hr.an.20170717.nc'
file5 = 'C:/Users/mp877190/Desktop/MET_Norway_current_data/11am/roms_nordic4_ZDEPTHS_hr.an.20170514.nc'
#read in netcdf file

dataset = Dataset(file3)
print(dataset.dimensions.keys())
print(dataset.variables.keys())

print(dataset.variables['u'])

#choose the right variables
x_velocity = np.array(dataset.variables['u'][0,0,:,:])
y_velocity = np.array(dataset.variables['v'][0,0,:,:])

plt.imshow(x_velocity)
plt.colorbar()
plt.clim(-0.3,0.3)
plt.title('x velocity (eastings)')
plt.show()

plt.imshow(y_velocity)
plt.colorbar()
plt.clim(-0.2,0.2)
plt.title('y velocity (northings)')
plt.show()

#caculate velocity 
x_squared = [[y**2 for y in x] for x in x_velocity]
y_squared = [[y**2 for y in x] for x in y_velocity]

sum_squared = np.zeros((9,10))
for i in range(len(x_squared)):
   # iterate through columns
   for j in range(len(x_squared[0])):
       sum_squared[i][j] = x_squared[i][j] + y_squared[i][j]
velocity = [[np.sqrt(y) for y in x] for x in sum_squared]


plt.imshow(sum_squared, cmap='pink')
plt.colorbar()
plt.clim(0,2)
plt.title('Sea currents for Heysham for 17/07/2017')
plt.show() #plt.savefig('C:/Users/mp877190/Desktop/seacurrents20170717.png')
