# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:53:31 2020

@author: VH Chen
"""
import numpy as np
from scotty.fun_general import read_floats_into_list_until
import math
from scipy import interpolate as interpolate
from scipy import constants as constants
import matplotlib.pyplot as plt
from scotty.fun_FFD import find_dpolflux_dR, find_dpolflux_dZ # For find_B if using efit files directly
from netCDF4 import Dataset














input_filename_suffix = '_29905_190'
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














delta_R = -0.01 #in the same units as data_R_coord
delta_Z = 0.01 #in the same units as data_Z_coord
delta_K_R = 0.1 #in the same units as K_R
delta_K_zeta = 0.1 #in the same units as K_zeta
delta_K_Z = 0.1 #in the same units as K_z


dataset = Dataset('efitOut_29905.nc')

time_index = 7
time_array = dataset.variables['time'][:]
#        equilibriumStatus_array = dataset.variables['equilibriumStatus'][:]

print(time_array[time_index])

output_group = dataset.groups['output']
#        input_group = dataset.groups['input']

profiles2D = output_group.groups['profiles2D']
data_unnormalised_poloidal_flux_grid = profiles2D.variables['poloidalFlux'][time_index][:][:] #unnormalised, as a function of R and Z
data_R_coord = profiles2D.variables['r'][time_index][:]
data_Z_coord = profiles2D.variables['z'][time_index][:]

radialProfiles = output_group.groups['radialProfiles']
Bt_array = radialProfiles.variables['Bt'][time_index][:]
r_array_B = radialProfiles.variables['r'][time_index][:]

separatrixGeometry = output_group.groups['separatrixGeometry']
geometricAxis = separatrixGeometry.variables['geometricAxis'][time_index] # R,Z location of the geometric axis

globalParameters = output_group.groups['globalParameters']
bvacRgeom = globalParameters.variables['bvacRgeom'][time_index] # Vacuum B field (= B_zeta, in vacuum) at the geometric axis

fluxFunctionProfiles = output_group.groups['fluxFunctionProfiles']
poloidalFlux = fluxFunctionProfiles.variables['normalizedPoloidalFlux'][:]
unnormalizedPoloidalFlux = fluxFunctionProfiles.variables['poloidalFlux'][time_index][:] # poloidalFlux as a function of normalised poloidal flux
rBphi = fluxFunctionProfiles.variables['rBphi'][time_index][:]

interp_rBphi = interpolate.interp1d(poloidalFlux, rBphi,
                                         kind='cubic', axis=-1, copy=True, bounds_error=True,
                                         fill_value=0, assume_sorted=False)

# normalised_polflux = polflux_const_m * poloidalFlux + polflux_const_c (think y = mx + c)
[polflux_const_m, polflux_const_c] = np.polyfit(unnormalizedPoloidalFlux,poloidalFlux,1) #linear fit
EFIT_poloidal_flux_grid = data_unnormalised_poloidal_flux_grid*polflux_const_m + polflux_const_c

interp_order = 3
interp_smoothing = 0
interp_poloidal_flux = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,EFIT_poloidal_flux_grid, bbox=[None, None, None, None], kx=interp_order, ky=interp_order, s=interp_smoothing)

# The last value of poloidalFlux for which rBphi (R * B_zeta) is specified. Outside this, we extrapolate B_zeta with 1/R
lastPoloidalFluxPoint = poloidalFlux[-1]

def find_B_R(q_R,q_Z,delta_R=delta_R,interp_poloidal_flux=interp_poloidal_flux,polflux_const_m=polflux_const_m):
    dpolflux_dZ = find_dpolflux_dZ(q_R,q_Z,delta_R,interp_poloidal_flux)
    B_R = -  dpolflux_dZ / (polflux_const_m * q_R)
    return B_R

def find_B_T(q_R,q_Z, lastPoloidalFluxPoint=lastPoloidalFluxPoint,bvacRgeom=bvacRgeom,geometricAxis=geometricAxis,interp_poloidal_flux=interp_poloidal_flux,interp_rBphi=interp_rBphi):
    polflux = interp_poloidal_flux(q_R,q_Z)
    if polflux <= lastPoloidalFluxPoint:
        # B_T from EFIT
        rBphi = interp_rBphi(polflux)
        B_T = rBphi/q_R
    else:
        # Extrapolate B_T
        # B_T = - B_vacuum * R_vacuum/q_R
        B_T = bvacRgeom * geometricAxis[0] / q_R
    return B_T

def find_B_Z(q_R,q_Z,delta_Z=delta_Z,interp_poloidal_flux=interp_poloidal_flux,polflux_const_m=polflux_const_m):
    dpolflux_dR = find_dpolflux_dR(q_R,q_Z,delta_Z,interp_poloidal_flux)
    B_Z = dpolflux_dR / (polflux_const_m * q_R)
    return B_Z

EFIT_B_R_grid = np.zeros_like(data_B_R_grid)
EFIT_B_T_grid = np.zeros_like(data_B_R_grid)
EFIT_B_Z_grid = np.zeros_like(data_B_R_grid)






for ii in range(0,len(data_R_coord)):
    for jj in range(0,len(data_Z_coord)):
        R_val = data_R_coord[ii]
        Z_val = data_Z_coord[jj]
        
        EFIT_B_R_grid[ii,jj] = find_B_R(R_val,Z_val)
        EFIT_B_T_grid[ii,jj] = find_B_T(R_val,Z_val)
        EFIT_B_Z_grid[ii,jj] = find_B_Z(R_val,Z_val)

plt.figure()
plt.subplot(2,3,1)
plt.title('B_R Torbeam')
plt.xlabel('R / m') # x-direction5
plt.ylabel('z / m')

contour_levels = np.linspace(0,1.0,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.xlim(data_R_coord[0],data_R_coord[-1])
plt.ylim(data_Z_coord[0],data_Z_coord[-1])

contour_scale = 0.2
contour_extent = min(abs(data_B_R_grid.max()),abs(data_B_R_grid.min()))
contour_levels = np.linspace(-contour_scale*contour_extent,contour_scale*contour_extent,21)
CS = plt.contourf(data_R_coord, data_Z_coord, np.transpose(data_B_R_grid),contour_levels,cmap='seismic')


plt.subplot(2,3,4)
plt.title('B_R EFIT')
plt.xlabel('R / m') # x-direction5
plt.ylabel('z / m')

contour_levels = np.linspace(0,1.0,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(EFIT_poloidal_flux_grid), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.xlim(data_R_coord[0],data_R_coord[-1])
plt.ylim(data_Z_coord[0],data_Z_coord[-1])

contour_scale = 0.2
contour_extent = min(abs(data_B_R_grid.max()),abs(data_B_R_grid.min()))
contour_levels = np.linspace(-contour_scale*contour_extent,contour_scale*contour_extent,21)
CS = plt.contourf(data_R_coord, data_Z_coord, np.transpose(EFIT_B_R_grid),contour_levels,cmap='seismic')


plt.subplot(2,3,2)
plt.title('B_Z Torbeam')
plt.xlabel('R / m') # x-direction5
plt.ylabel('z / m')

contour_levels = np.linspace(0,1.0,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.xlim(data_R_coord[0],data_R_coord[-1])
plt.ylim(data_Z_coord[0],data_Z_coord[-1])

contour_scale = 0.5
contour_extent = min(abs(data_B_Z_grid.max()),abs(data_B_Z_grid.min()))
contour_levels = np.linspace(-contour_scale*contour_extent,contour_scale*contour_extent,21)
CS = plt.contourf(data_R_coord, data_Z_coord, np.transpose(data_B_Z_grid),contour_levels,cmap='seismic')


plt.subplot(2,3,5)
plt.title('B_Z EFIT')
plt.xlabel('R / m') # x-direction5
plt.ylabel('z / m')

contour_levels = np.linspace(0,1.0,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(EFIT_poloidal_flux_grid), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.xlim(data_R_coord[0],data_R_coord[-1])
plt.ylim(data_Z_coord[0],data_Z_coord[-1])

contour_scale = 0.5
contour_extent = min(abs(data_B_Z_grid.max()),abs(data_B_Z_grid.min()))
contour_levels = np.linspace(-contour_scale*contour_extent,contour_scale*contour_extent,21)
CS = plt.contourf(data_R_coord, data_Z_coord, np.transpose(EFIT_B_Z_grid),contour_levels,cmap='seismic')



plt.subplot(2,3,3)
plt.title('B_T Torbeam')
plt.xlabel('R / m') # x-direction5
plt.ylabel('z / m')

contour_levels = np.linspace(0,1.0,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.xlim(data_R_coord[0],data_R_coord[-1])
plt.ylim(data_Z_coord[0],data_Z_coord[-1])

contour_scale = 0.1
contour_levels = np.linspace(contour_scale*data_B_T_grid.min(),data_B_T_grid.max()*(1.0+contour_scale),21)
CS = plt.contourf(data_R_coord, data_Z_coord, np.transpose(data_B_T_grid),contour_levels,cmap='plasma')


plt.subplot(2,3,6)
plt.title('B_T EFIT')
plt.xlabel('R / m') # x-direction5
plt.ylabel('z / m')

contour_levels = np.linspace(0,1.0,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(EFIT_poloidal_flux_grid), contour_levels,vmin=0,vmax=1,cmap='plasma_r')
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.xlim(data_R_coord[0],data_R_coord[-1])
plt.ylim(data_Z_coord[0],data_Z_coord[-1])

contour_scale = 0.1
contour_levels = np.linspace(contour_scale*data_B_T_grid.min(),data_B_T_grid.max()*(1.0+contour_scale),21)
CS = plt.contourf(data_R_coord, data_Z_coord, np.transpose(EFIT_B_T_grid),contour_levels,cmap='plasma')


