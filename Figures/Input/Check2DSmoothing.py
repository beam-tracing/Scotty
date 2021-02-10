# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:16:49 2020

@author: VH Chen
"""

import numpy as np
# from Scotty_fun_general import read_floats_into_list_until
import math
from scipy import interpolate as interpolate
from scipy import constants as constants
import matplotlib.pyplot as plt
import tikzplotlib


input_filename_suffix = '_29910_190'
launch_freq_GHz = 55.0

input_files_path ='D:\\Dropbox\\VHChen2020\\Data\\Input_Files_29Apr2019\\'
#input_files_path ='D:\\Dropbox\\VHChen2018\\Data\\Input_Files_29Apr2019\\'
#input_files_path ='D:\\Dropbox\\VHChen2019\\Code - Scotty\\Benchmark_9\\Torbeam\\'
#input_files_path = os.path.dirname(os.path.abspath(__file__)) + '\\'

def read_floats_into_list_until(terminator, lines):
    # Reads the lines of a file until the string (terminator) is read
    # Currently used to read topfile
    # Written by NE Bricknell
    lst = []
    while True:
        try: line = lines.readline()
        except StopIteration: break
        if terminator in line: break
        elif not line: break
        lst.extend(map(float,  line.split()))
    return lst


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
order = 5
smoothing=3
interp_B_R = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_B_R_grid, bbox=[None, None, None, None], kx=order, ky=order, s=smoothing)
interp_B_T = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_B_T_grid, bbox=[None, None, None, None], kx=order, ky=order, s=smoothing)
interp_B_Z = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_B_Z_grid, bbox=[None, None, None, None], kx=order, ky=order, s=smoothing)

interp_poloidal_flux = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_poloidal_flux_grid, bbox=[None, None, None, None], kx=order, ky=order, s=smoothing)
#    interp_poloidal_flux = interpolate.interp2d(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), kind='cubic',
#                                           copy=True, bounds_error=False, fill_value=None) # Flux extrapolated outside region

interp_density_1D = interpolate.interp1d(ne_data_poloidal_flux_array, ne_data_density_array,
                                         kind='cubic', axis=-1, copy=True, bounds_error=False,
                                         fill_value=0, assume_sorted=False) # density is 0 outside the LCFS, hence the fill_value

numberOfInterpPoints = 129
interpolated_R_coord = np.linspace(data_R_coord[0], data_R_coord[-1], numberOfInterpPoints)
interpolated_Z_coord = np.linspace(data_Z_coord[0], data_Z_coord[-1], numberOfInterpPoints)
interpolated_B_R = interp_B_R(interpolated_R_coord,interpolated_Z_coord)
interpolated_B_T = interp_B_T(interpolated_R_coord,interpolated_Z_coord)
interpolated_B_Z = interp_B_Z(interpolated_R_coord,interpolated_Z_coord)
interpolated_poloidal_flux = interp_poloidal_flux(interpolated_R_coord,interpolated_Z_coord)

start_point_R = 10
start_point_Z = 25

end_point_R = 90
end_point_Z = 110

plt.figure()
plt.title('Rz')
plt.xlabel('R / m') # x-direction5
plt.ylabel('z / m')
    
contour_levels = np.linspace(0.1,1.0,10)
CS = plt.contour(data_R_coord[start_point_R:end_point_R], data_Z_coord[start_point_Z:end_point_Z], np.transpose(data_poloidal_flux_grid[start_point_R:end_point_R,start_point_Z:end_point_Z]), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.xlim(data_R_coord[start_point_R],data_R_coord[end_point_R])
plt.ylim(data_Z_coord[start_point_Z],data_Z_coord[end_point_Z])
tikzplotlib.save("data.tex")

plt.figure()
plt.title('Rz')
plt.xlabel('R / m') # x-direction5
plt.ylabel('z / m')
    
contour_levels = np.linspace(0.1,1.0,10)
CS = plt.contour(interpolated_R_coord[start_point_R:end_point_R], interpolated_Z_coord[start_point_Z:end_point_Z], np.transpose(interpolated_poloidal_flux[start_point_R:end_point_R,start_point_Z:end_point_Z]), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.xlim(interpolated_R_coord[start_point_R],interpolated_R_coord[end_point_R])
plt.ylim(interpolated_Z_coord[start_point_Z],interpolated_Z_coord[end_point_Z])
tikzplotlib.save("interp_and_smooth.tex")

plt.figure()
plt.title('Rz')
plt.xlabel('R / m') # x-direction5
plt.ylabel('z / m')
    
contour_levels = np.linspace(0,1.0,101)
CS = plt.contourf(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid),contour_levels,vmin=0,vmax=1,cmap='inferno_r')
#plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.xlim(data_R_coord[0],data_R_coord[-1])
plt.ylim(data_Z_coord[0],data_Z_coord[-1])

plt.figure()
plt.title('Rz')
plt.xlabel('R / m') # x-direction5
plt.ylabel('z / m')
    
contour_levels = np.linspace(0,1.0,101)
CS = plt.contourf(interpolated_R_coord, interpolated_Z_coord, np.transpose(interpolated_poloidal_flux), contour_levels,vmin=0,vmax=1,cmap='inferno_r')
plt.xlim(interpolated_R_coord[0],interpolated_R_coord[-1])
plt.ylim(interpolated_Z_coord[0],interpolated_Z_coord[-1])


