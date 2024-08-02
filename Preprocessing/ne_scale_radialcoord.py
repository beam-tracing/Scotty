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
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

#def find_zero(C_1,C_2,C_3,C_4):
#    polflux_lower=0.5
#    polflux_upper=1.5
#    
#    fun_lower=linear_times_tanh(polflux_lower,C_1,C_2,C_3,C_4)
#    fun_upper=linear_times_tanh(polflux_upper,C_1,C_2,C_3,C_4)
#    
#        while (() and ()):
#    return polflux_zero_guess

def linear_times_tanh(polflux,C_1,C_2,C_3,C_4):
    return (C_1*polflux + C_2)*np.tanh(C_3 * polflux + C_4)

def linear_times_tanh_constrained(polflux,C_1,C_3,C_4):
    C_2 = -C_1
    return (C_1*polflux + C_2)*np.tanh(C_3 * polflux + C_4)

#input_files_path ='D:\\Dropbox\\VHChen2018\\Data\\Input_Files_08Apr2019\\'
#torbeam_directory_path = 'D:\\Dropbox\\VHChen2018\\Code - Torbeam\\torbeam_ccfe_val_test\\'
#torbeam_directory_path = 'C:\\Users\\chenv\\Dropbox\\VHChen2018\\Code - Torbeam\\torbeam_ccfe_val_test\\'
#input_files_path ='C:\\Users\\chenv\\Dropbox\\VHChen2018\\Data\\Input_Files_29Apr2019\\'
#torbeam_directory_path = 'C:\\Users\\chenv\\Dropbox\\VHChen2018\\Data\\Input_Files_29Apr2019\\'
# input_files_path ='D:\\Dropbox\\VHChen2020\\Data\\Input_Files_29Apr2019\\'
# torbeam_directory_path = 'D:\\Dropbox\\VHChen2020\\Data\\Input_Files_29Apr2019\\'
input_files_path ='C:\\Dropbox\\VHChen2023\\Data - Equilibrium\\'
torbeam_directory_path = 'C:\\Dropbox\\VHChen2023\\Data - Equilibrium\\'

suffix = '_188842_2500ms'
ne_filename = input_files_path + 'ne' +suffix+ '.dat'

ne_data = np.fromfile(ne_filename,dtype=float, sep='   ') # electron density as a function of poloidal flux label

ne_data_length = int(ne_data[0])
ne_data_density_array = ne_data[2::2] # in units of 10.0**19 m-3
ne_data_radialcoord_array = ne_data[1::2]   
ne_data_flux_array = ne_data_radialcoord_array**2
# ne_data_flux_array = ne_data[1::2] # My new IDL file outputs the flux density directly, instead of radialcoord

scaling = 0.82

# Loading from DIII-D
ne_data_density_array = ne_data_density_array / 10**19
##


plt.figure()
plt.scatter(ne_data_flux_array,ne_data_density_array,color='black')
plt.ylim([0, 6])
plt.xlim([0, 1.5])


ne_flux_output = ne_data_flux_array
ne_data_output = scaling*ne_data_density_array

plt.plot(ne_flux_output, ne_data_output)

radialcoord_output = np.sqrt(ne_flux_output)
output_length = len(radialcoord_output)

ne_data_file = open(torbeam_directory_path + 'ne' +suffix+ '_scaled.dat','w')  
ne_data_file.write(str(int(output_length)) + '\n') 
for ii in range(0, output_length):
    ne_data_file.write('{:.8e} {:.8e} \n'.format(radialcoord_output[ii],ne_data_output[ii]))        
ne_data_file.close() 