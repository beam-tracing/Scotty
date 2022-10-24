# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:16:49 2020

@author: VH Chen
"""

import numpy as np
from scotty.fun_general import read_floats_into_list_until
import math
from scipy import interpolate as interpolate
from scipy import constants as constants
import matplotlib.pyplot as plt

input_filename_suffix = '_29910_190'
launch_freq_GHz = 55.0

input_files_path ='D:\\Dropbox\\VHChen2020\\Data\\Input_Files_29Apr2019\\'
#input_files_path ='D:\\Dropbox\\VHChen2018\\Data\\Input_Files_29Apr2019\\'
#input_files_path ='D:\\Dropbox\\VHChen2019\\Code - Scotty\\Benchmark_9\\Torbeam\\'
#input_files_path = os.path.dirname(os.path.abspath(__file__)) + '\\'



# Importing data from input files
# ne.dat, topfile
# Others: inbeam.dat, Te.dat (not currently used in this code)
# ne_filename = input_files_path + 'ne' +input_filename_suffix+ '_smoothed.dat'
ne_filename = input_files_path + 'ne' +input_filename_suffix+ '_fitted.dat'
#ne_filename = input_files_path + 'ne' +input_filename_suffix+ '.dat'

topfile_filename = input_files_path + 'topfile' +input_filename_suffix


ne_data = np.fromfile(ne_filename,dtype=float, sep='   ') # electron density as a function of poloidal flux label
with open(topfile_filename) as f:
    while not 'X-coordinates' in f.readline(): pass # Start reading only from X-coords onwards
    data_R_coord = read_floats_into_list_until('Z-coordinates', f)
    data_Z_coord = read_floats_into_list_until('B_R', f)
    data_B_R_grid = read_floats_into_list_until('B_t', f)
    data_B_T_grid = read_floats_into_list_until('B_Z', f)
    data_B_Z_grid = read_floats_into_list_until('psi', f)
    data_poloidal_flux_grid = read_floats_into_list_until('you fall asleep', f)
# ------------------------------

# Tidying up the input data
launch_angular_frequency = 2*math.pi*10.0**9 * launch_freq_GHz
wavenumber_K0 = launch_angular_frequency / constants.c

ne_data_length = int(ne_data[0])
ne_data_density_array = 1.1*ne_data[2::2] # in units of 10.0**19 m-3
print('Warninig: Scale factor of 1.05 used')
ne_data_radialcoord_array = ne_data[1::2]
ne_data_poloidal_flux_array = ne_data_radialcoord_array**2 # Loading radial coord for now, makes it easier to benchmark with Torbeam. Hence, have to convert to poloidal flux
#ne_data_poloidal_flux_array = ne_data[1::2] # My new IDL file outputs the flux density directly, instead of radialcoord

data_B_R_grid = np.transpose((np.asarray(data_B_R_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
data_B_T_grid = np.transpose((np.asarray(data_B_T_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
data_B_Z_grid = np.transpose((np.asarray(data_B_Z_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
data_poloidal_flux_grid = np.transpose((np.asarray(data_poloidal_flux_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
# -------------------

# Interpolation functions declared
order = 3
smoothing=2
interp_B_R = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_B_R_grid, bbox=[None, None, None, None], kx=order, ky=order, s=smoothing)
interp_B_T = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_B_T_grid, bbox=[None, None, None, None], kx=order, ky=order, s=smoothing)
interp_B_Z = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_B_Z_grid, bbox=[None, None, None, None], kx=order, ky=order, s=smoothing)

interp_poloidal_flux = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_poloidal_flux_grid, bbox=[None, None, None, None], kx=order, ky=order, s=smoothing)
#    interp_poloidal_flux = interpolate.interp2d(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), kind='cubic',
#                                           copy=True, bounds_error=False, fill_value=None) # Flux extrapolated outside region

interp_density_1D = interpolate.interp1d(ne_data_poloidal_flux_array, ne_data_density_array,
                                         kind='cubic', axis=-1, copy=True, bounds_error=False,
                                         fill_value=0, assume_sorted=False) # density is 0 outside the LCFS, hence the fill_value

numberOfInterpPoints = 1000
interpolated_R_coord = np.linspace(data_R_coord[0], data_R_coord[-1], numberOfInterpPoints)
interpolated_Z_coord = np.linspace(data_Z_coord[0], data_Z_coord[-1], 2*numberOfInterpPoints)
interpolated_B_R = interp_B_R(interpolated_R_coord,interpolated_Z_coord)
interpolated_B_T = interp_B_T(interpolated_R_coord,interpolated_Z_coord)
interpolated_B_Z = interp_B_Z(interpolated_R_coord,interpolated_Z_coord)
interpolated_poloidal_flux = interp_poloidal_flux(interpolated_R_coord,interpolated_Z_coord)

interpolated_poloidal_flux2 = np.zeros([numberOfInterpPoints,2*numberOfInterpPoints])

plt.figure()
plt.title('Rz')
plt.xlabel('R / m') # x-direction5
plt.ylabel('z / m')
    
contour_levels = np.linspace(0,1.0,11)
CS = plt.contour(interpolated_R_coord, interpolated_Z_coord, np.transpose(interpolated_poloidal_flux), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.xlim(interpolated_R_coord[0],interpolated_R_coord[-1])
plt.ylim(interpolated_Z_coord[0],interpolated_Z_coord[-1])

plt.figure()
plt.title('Rz')
plt.xlabel('R / m') # x-direction5
plt.ylabel('z / m')
    
contour_levels = np.linspace(0,1.0,101)
CS = plt.contourf(interpolated_R_coord, interpolated_Z_coord, np.transpose(interpolated_poloidal_flux),contour_levels,vmin=0,vmax=1,cmap='inferno_r')
#plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.xlim(interpolated_R_coord[0],interpolated_R_coord[-1])
plt.ylim(interpolated_Z_coord[0],interpolated_Z_coord[-1])










#
B_magnitude_grid = np.sqrt(data_B_R_grid**2 + data_B_T_grid**2 + data_B_Z_grid**2)
grad_B_magnitude_grid = np.gradient(B_magnitude_grid,data_R_coord,data_Z_coord)
#grad_grad_B_magnitude_grid = np.gradient(grad_B_magnitude_grid,data_R_coord,data_Z_coord)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title('|B|')
plt.xlabel('R / m') # x-direction5
plt.ylabel('z / m')
    
contour_levels = np.linspace(0,1.0,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces

contour_scale = 0.2
contour_levels = np.linspace(contour_scale*B_magnitude_grid.min(),contour_scale*B_magnitude_grid.max(),11)
CS = plt.contourf(data_R_coord, data_Z_coord, np.transpose(B_magnitude_grid),contour_levels,cmap='inferno')
#plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.xlim(interpolated_R_coord[0],interpolated_R_coord[-1])
plt.ylim(interpolated_Z_coord[0],interpolated_Z_coord[-1])
plt.colorbar()

plt.subplot(1,3,2)
plt.title('grad |B|')
plt.xlabel('R / m') # x-direction5
plt.ylabel('z / m')
    
contour_levels = np.linspace(0,1.0,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces

contour_scale = 1.0
contour_levels = np.linspace(contour_scale*grad_B_magnitude_grid.min(),contour_scale*grad_B_magnitude_grid.max(),11)
CS = plt.contourf(data_R_coord, data_Z_coord, np.transpose(grad_B_magnitude_grid),contour_levels,cmap='inferno')
#plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.xlim(interpolated_R_coord[0],interpolated_R_coord[-1])
plt.ylim(interpolated_Z_coord[0],interpolated_Z_coord[-1])
plt.subplot(1,3,3)

# plt.title('grad grad |B|')
# plt.xlabel('R / m') # x-direction5
# plt.ylabel('z / m')
    
# contour_levels = np.linspace(0,1.0,11)
# CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
# plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces

# contour_scale = 1.0
# contour_levels = np.linspace(contour_scale*grad_grad_B_magnitude_grid.min(),contour_scale*grad_grad_B_magnitude_grid.max(),11)
# CS = plt.contourf(data_R_coord, data_Z_coord, np.transpose(grad_grad_B_magnitude_grid),contour_levels,cmap='inferno')
# #plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
# plt.xlim(interpolated_R_coord[0],interpolated_R_coord[-1])
# plt.ylim(interpolated_Z_coord[0],interpolated_Z_coord[-1])

#for ii in range(0,numberOfInterpPoints):
#    for jj in range(0,numberOfInterpPoints):
#        interpolated_poloidal_flux2[ii,jj] = 

#plt.figure()
#plt.title('Rz')
#plt.xlabel('R / m') # x-direction5
#plt.ylabel('z / m')
#    
#contour_levels = np.linspace(0,1.0,11)
#CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
#plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
#plt.xlim(data_R_coord[0],data_R_coord[-1])
#plt.ylim(data_Z_coord[0],data_Z_coord[-1])
#

#
#
#plt.figure()
#plt.title('Rz')
#plt.xlabel('R / m') # x-direction5
#plt.ylabel('z / m')
#    
#contour_levels = np.linspace(0,1.0,11)
#CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
#plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
#
#contour_scale = 0.2
#contour_levels = np.linspace(contour_scale*interpolated_B_R.min(),contour_scale*interpolated_B_R.max(),11)
#CS = plt.contourf(interpolated_R_coord, interpolated_Z_coord, np.transpose(interpolated_B_R),contour_levels,cmap='seismic')
##plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
#plt.xlim(interpolated_R_coord[0],interpolated_R_coord[-1])
#plt.ylim(interpolated_Z_coord[0],interpolated_Z_coord[-1])
#
#plt.figure()
#plt.title('Rz')
#plt.xlabel('R / m') # x-direction5
#plt.ylabel('z / m')
#    
#contour_levels = np.linspace(0,1.0,11)
#CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
#plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
#
#contour_scale = 0.2
#contour_levels = np.linspace(contour_scale*interpolated_B_Z.min(),contour_scale*interpolated_B_Z.max(),11)
#CS = plt.contourf(interpolated_R_coord, interpolated_Z_coord, np.transpose(interpolated_B_Z),contour_levels,cmap='seismic')
##plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
#plt.xlim(interpolated_R_coord[0],interpolated_R_coord[-1])
#plt.ylim(interpolated_Z_coord[0],interpolated_Z_coord[-1])