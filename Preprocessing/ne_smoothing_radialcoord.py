# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 05:54:07 2019

Let's try smoothing out the electron density data

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interpolate
from scipy import signal as signal

#input_files_path ='D:\\Dropbox\\VHChen2018\\Data\\Input_Files_08Apr2019\\'
#torbeam_directory_path = 'D:\\Dropbox\\VHChen2018\\Code - Torbeam\\torbeam_ccfe_val_test\\'
#torbeam_directory_path = 'C:\\Users\\chenv\\Dropbox\\VHChen2018\\Code - Torbeam\\torbeam_ccfe_val_test\\'
#input_files_path ='C:\\Users\\chenv\\Dropbox\\VHChen2018\\Data\\Input_Files_29Apr2019\\'
#torbeam_directory_path = 'C:\\Users\\chenv\\Dropbox\\VHChen2018\\Data\\Input_Files_29Apr2019\\'
input_files_path ='D:\\Dropbox\\VHChen2018\\Data\\Input_Files_29Apr2019\\'
torbeam_directory_path = 'D:\\Dropbox\\VHChen2018\\Data\\Input_Files_29Apr2019\\'

suffix = '_29905_200'
ne_filename = input_files_path + 'ne' +suffix+ '.dat'

ne_data = np.fromfile(ne_filename,dtype=float, sep='   ') # electron density as a function of poloidal flux label

ne_data_length = int(ne_data[0])
ne_data_density_array = ne_data[2::2] # in units of 10.0**19 m-3
#ne_data_radialcoord_array = ne_data[1::2]   
#ne_data_flux_array = ne_data_radialcoord_array**2
ne_data_flux_array = ne_data[1::2] # My new IDL file outputs the flux density directly, instead of radialcoord

interp_length = 2001
psi_end = 2.0
flux_interp_array = np.linspace(0,psi_end,interp_length)

window_length_data = 11
polyorder_data = 3
ne_data_density_array_smoothed = signal.savgol_filter(ne_data_density_array, 
                                                      window_length_data, polyorder_data, deriv=0, delta=1.0, 
                                                      axis=-1, mode='interp', cval=0.0) 

core_density = np.mean(ne_data_density_array[0:10])
interp_cubic = interpolate.interp1d(ne_data_flux_array, ne_data_density_array,
                                    fill_value=(core_density,0), kind='cubic', bounds_error=False)
density_interp_array = interp_cubic(flux_interp_array)
interp_cubic2 = interpolate.interp1d(ne_data_flux_array, 1.05*ne_data_density_array_smoothed,
                                    fill_value=(core_density,0), kind='cubic', bounds_error=False)
density_interp_array2 = interp_cubic2(flux_interp_array)

window_length_interp = 401
polyorder_interp = 3
density_smoothed_array = signal.savgol_filter(density_interp_array, window_length_interp, 
                                              polyorder_interp, deriv=0, delta=1.0, 
                                              axis=-1, mode='interp', cval=0.0) 
density_smoothed_array[density_smoothed_array<0] = 0

density_smoothed_array2 = signal.savgol_filter(density_interp_array2, window_length_interp, 
                                              polyorder_interp, deriv=0, delta=1.0, 
                                              axis=-1, mode='interp', cval=0.0) 
density_smoothed_array2[density_smoothed_array2<0] = 0
#density_interp_array = interp_cubic(flux_interp_array)

plt.figure()
plt.scatter(ne_data_flux_array,ne_data_density_array,color='black')
plt.scatter(ne_data_flux_array,ne_data_density_array_smoothed,color='red')
plt.plot(flux_interp_array,density_smoothed_array,color='black')
plt.plot(flux_interp_array,density_smoothed_array2,color='red')

radialcoord_interp_array = np.sqrt(flux_interp_array)

ne_data_file = open(torbeam_directory_path + 'ne' +suffix+ '_smoothed.dat','w')  
ne_data_file.write(str(int(interp_length)) + '\n') 
for ii in range(0, interp_length):
    ne_data_file.write('{:.8e} {:.8e} \n'.format(radialcoord_interp_array[ii],density_smoothed_array2[ii]))        
ne_data_file.close() 