# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:37:48 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import tikzplotlib
from netCDF4 import Dataset
from scipy import interpolate
from scipy.optimize import curve_fit

def find_nearest(array,  value): #returns the index
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)

def linear_times_tanh(polflux,C_1,C_2,C_3,C_4):
    return (C_1*polflux + C_2)*np.tanh(C_3 * polflux + C_4)

def linear_plus_tanh(polflux,C_1,C_2,C_3,C_4,C_5):
    return (C_1*polflux + C_2) + C_3 * np.tanh(C_4 * (polflux - C_5) ) 

def tanh(polflux,C_1,C_2,C_3):
    return C_1 * np.tanh(C_2 * (polflux - C_3) ) 

def mtanh(x,b_slope):
    return ((1+b_slope*x)*np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def process_Thomson(shot, time):
    # Path of the equilibrium data files
    eq_file_path = 'D:\\Dropbox\\VHChen2020\\Code - DBS\\Equilibrium\\'

    filename = eq_file_path + str(shot) + '_equilibrium_data.npz '
    loadfile = np.load(filename)
    # Thomson data
    t_TSC = loadfile['t_TSC']
    R_TSC = loadfile['R_TSC'] # in meters [numberOfTimePoints,numberOfRPointsPerTimePoint]
    Z_TSC = 0.015 # From Rory Scannell
    ne_TSC = loadfile['ne_TSC']
    Te_TSC = loadfile['Te_TSC']
    loadfile.close()

    t_index_Thomson = find_nearest(t_TSC,  time)
    t_nearest_Thomson = t_TSC[t_index_Thomson]
    
    EFITpp_file_path = 'D:\\Dropbox\\VHChen2020\\Data\\MSE_efitruns\\' + str(shot) + '\\'
    dataset = Dataset(EFITpp_file_path + 'efitOut.nc')
    t_EFIT = dataset.variables['time'][:] 
    print(t_EFIT)
    t_index_EFIT = find_nearest(t_EFIT,  time)
    t_nearest_EFIT = t_EFIT[t_index_EFIT]

    output_group = dataset.groups['output']
    
    fluxFunctionProfiles = output_group.groups['fluxFunctionProfiles']
    poloidalFlux = fluxFunctionProfiles.variables['normalizedPoloidalFlux'][:]
    
    profiles2D = output_group.groups['profiles2D']
    data_unnormalised_poloidal_flux_grid = profiles2D.variables['poloidalFlux'][t_index_EFIT][:][:] #unnormalised, as a function of R and Z
    data_R_coord = profiles2D.variables['r'][t_index_EFIT][:]
    data_Z_coord = profiles2D.variables['z'][t_index_EFIT][:]

    fluxFunctionProfiles = output_group.groups['fluxFunctionProfiles']
    unnormalizedPoloidalFlux = fluxFunctionProfiles.variables['poloidalFlux'][t_index_EFIT][:] # poloidalFlux as a function of normalised poloidal flux

    dataset.close()    
  
    # normalised_polflux = polflux_const_m * poloidalFlux + polflux_const_c (think y = mx + c)        
    [polflux_const_m, polflux_const_c] = np.polyfit(unnormalizedPoloidalFlux,poloidalFlux,1) #linear fit
    data_poloidal_flux_grid = data_unnormalised_poloidal_flux_grid*polflux_const_m + polflux_const_c
    
    interp_order = 3 # For the 2D interpolation functions
    interp_smoothing = 0 # For the 2D interpolation functions. For no smoothing, set to 0    
    interp_poloidal_flux = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_poloidal_flux_grid, bbox=[None, None, None, None], kx=interp_order, ky=interp_order, s=interp_smoothing)
        
    print('Time: requested:', "{:.3g}".format(time),'s' )    
    print('Time: Thomson data:', "{:.3g}".format(t_nearest_Thomson),'s' )
    print('Time: EFIT data:', "{:.3g}".format(t_nearest_EFIT),'s' )
    print(t_index_Thomson)
    
    Thomson_R_coord = R_TSC[t_index_Thomson,:]
    Thomson_Z_coord = Z_TSC * np.ones_like(Thomson_R_coord)
    
    Thomson_poloidal_flux = interp_poloidal_flux(Thomson_R_coord,Thomson_Z_coord,grid=False) 

    return Thomson_poloidal_flux[:], ne_TSC[t_index_Thomson,:]

shot_array1 = np.array([29904, 29905, 29906, 29908, 29909, 29910])
shot_array2A = np.array([29677, 29678, 29679, 29681, 29683, 29684])
shot_array2B = np.array([29692, 29693])
shot_array2  = np.append(shot_array2A,shot_array2B)
shot_array = np.array([29904])


shot_array_check = np.array([29677, 29683, 29684])




counter = 0

pol_flux, ne = process_Thomson(29908, 0.19)

pol_flux_noNaN = pol_flux[~np.isnan(ne)]
ne_noNaN = ne[~np.isnan(ne)] / 10**19

pol_flux_selected = pol_flux_noNaN[np.argwhere(pol_flux_noNaN>0.5).flatten()]
ne_selected = ne_noNaN[np.argwhere(pol_flux_noNaN>0.5).flatten()]



pol_flux_plot = np.linspace(0,2.0,101)
# ne_plot = linear_times_tanh(pol_flux_plot,0,1,-1,-1.2)

# 3.5,-2.1,1.22
# 3.25,-2.4,1.22

plt.figure()
plt.plot(pol_flux_noNaN[pol_flux.argmin():],ne_noNaN[pol_flux.argmin():],'b',linestyle='',marker='o',label='outboard')
plt.plot(pol_flux_noNaN[:pol_flux.argmin()],ne_noNaN[:pol_flux.argmin()],'r',linestyle='',marker='^',label='inboard')
plt.plot(pol_flux_plot,tanh(pol_flux_plot,3.5,-2.1,1.22),'k',label='fit')
# plt.plot(pol_flux_plot, (popt[0]*pol_flux_plot + popt[1]))
# plt.plot(pol_flux_plot, popt[2] * np.tanh(popt[3] * (pol_flux_plot - popt[4])) )
plt.ylim([0, 3.5])
plt.xlim([0, 1.4])
plt.ylabel('n_e / 10^19 m-3') # x-direction
plt.xlabel(r'$\psi_p$')
plt.title('Density profile')
plt.legend(loc='lower left',edgecolor='k')
tikzplotlib.save("density_profile.tex")


