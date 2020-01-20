import numpy as np
import math
import copy

import pdb

#distance function here
def distance(lat_image, lon_image, grid_min_lat, grid_min_lon, pp, qq, res):
	"""
	This function calculates the distance of the original image points 
	from the centre on the common grid cell in meters. 
	Return: array of distances of satellite pixels to the common grid cells.
	"""
	#calculate lat and lon for p and q
	lat_pq = grid_min_lat + ((qq - (1/2.0))/res)
	lon_pq = grid_min_lon + ((pp - (1/2.0))/res)
	# calculate the distance as
	delta_lat = (lat_image - lat_pq) * 180 / np.pi
	delta_lon = (lon_image - lon_pq) * 180 / np.pi
	# define radius of Earth
	r_earth = 6371000 #m
	# km distance expressed in radians
	lat_image_rad = lat_image * 180 / np.pi
	distance = np.sqrt((delta_lat)**2 + (np.cos(lat_image_rad)*delta_lon)**2) * r_earth
	
	return distance

def regridding(image_data, min_lon, min_lat, spacing, shape_common_grid):
	"""
	This function takes in the 3D matrix of the satellite data, as well as 
	the minimum values of the latitude and longitude, the spacing and shape of common grid.
	For each band it creates an empty array which has the size od the common grid.
	It creates indices for lat and lon from the original satellite image.
	Then it populates the common grid with the pixels from original image
	using nearest neighbour (distance) function.
	Return: 3D array of regridded satellite data (only bands, no lat and lon), counter and distance.
	"""
	# create an empty m x n array for each channel
	band_data = np.zeros((shape_common_grid)) ####define for each band
	band_data = band_data[0,:,:]
	band1_data = copy.copy(band_data) #band_data[0,:,:]
	band2_data = copy.copy(band_data) #band_data[0,:,:]
	band3_data = copy.copy(band_data) #band_data[0,:,:]
	band4_data = copy.copy(band_data) #band_data[0,:,:]
	band5_data = copy.copy(band_data) #band_data[0,:,:]
	band6_data = copy.copy(band_data) #band_data[0,:,:]
	band7_data = copy.copy(band_data) #band_data[0,:,:]
	#band8_data = copy.copy(band_data) #band_data[0,:,:]
	band9_data = copy.copy(band_data) #band_data[0,:,:]
	band10_data = copy.copy(band_data) #band_data[0,:,:]
	band11_data = copy.copy(band_data) #band_data[0,:,:]
	
	# a count array of the same size
	C = np.zeros((shape_common_grid),dtype=np.int) ### this only one
	C = C[0,:,:]
	# a distance array
	D = np.zeros((shape_common_grid))
	D = D[0,:,:]

	# take arrays of full resolution input
	im_lat = image_data[0,:,:]
	im_lon = image_data[1,:,:]
	data1 = image_data[2,:,:]
	data2 = image_data[3,:,:]
	data3 = image_data[4,:,:]
	data4 = image_data[5,:,:]
	data5 = image_data[6,:,:]
	data6 = image_data[7,:,:]
	data7 = image_data[8,:,:]
	#data8 = image_data[9,:,:]
	data9 = image_data[9,:,:]
	data10 = image_data[10,:,:]
	data11 = image_data[11,:,:]
	
	# transform lat and lon arrays
	# by subtracting the minimum value from the common grid
	# and dividing by spacing of common grid
	lat_transf = (im_lat - min_lat) / spacing
	lon_transf = (im_lon - min_lon) / spacing
	# round down the values from transf arrays
	lat_rounded = np.floor(lat_transf)
	lon_rounded = np.floor(lon_transf)
	print("lat_rounded", lat_rounded)
	print("lon_rounded", lon_rounded)
	# index of the original image lat and lon 
	
	# go through entire x and y for image data
	# see if they are all positive integers
	# 0 is a valid number

	for (i,j), q in np.ndenumerate(lat_rounded):
		i = int(i)
		j = int(j)
		p = int(lon_rounded[i,j])
		q = int(lat_rounded[i,j])

		if q >= 0 and q <= 400 and p >=0 and p <= 400:
			if C[p,q] == 0:
				band1_data[p,q] = data1[i,j]
				band2_data[p,q] = data2[i,j]
				band3_data[p,q] = data3[i,j]
				band4_data[p,q] = data4[i,j]
				band5_data[p,q] = data5[i,j]
				band6_data[p,q] = data6[i,j]
				band7_data[p,q] = data7[i,j]
				#band8_data[p,q] = data8[i,j]
				band9_data[p,q] = data9[i,j]
				band10_data[p,q] = data10[i,j]
				band11_data[p,q] = data11[i,j]
				D[p,q] = distance(im_lat[i,j], im_lon[i,j], min_lat, min_lon, p, q, spacing)
				C[p,q] = 1
				#C[p,q] += 1
			else:
				d = distance(im_lat[i,j], im_lon[i,j], min_lat, min_lon, p, q, spacing)
				if d < D[p,q]:
					band1_data[p,q] = data1[i,j]
					band2_data[p,q] = data2[i,j]
					band3_data[p,q] = data3[i,j]
					band4_data[p,q] = data4[i,j]
					band5_data[p,q] = data5[i,j]
					band6_data[p,q] = data6[i,j]
					band7_data[p,q] = data7[i,j]
					#band8_data[p,q] = data8[i,j]
					band9_data[p,q] = data9[i,j]
					band10_data[p,q] = data10[i,j]
					band11_data[p,q] = data11[i,j]
					D[p,q] = d
		#else:
			#print("p and q out of range")     #### later can print p and q values
	return np.concatenate([[band1_data], [band2_data], [band3_data], [band4_data], [band5_data], [band6_data], [band7_data], [band9_data], [band10_data], [band11_data]]), C, D
