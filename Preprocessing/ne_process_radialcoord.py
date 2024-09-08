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


def interp_smooth_Thomson(pol_flux_to_use, ne_to_use,ne_err_to_use):
    # Choosing 's' is important
    # values of s used previously: 27 (160ms, 190ms), 44 (220ms), 59 (250ms)
    interp_ne = interpolate.UnivariateSpline(pol_flux_to_use, ne_to_use,
                                              w=None, bbox=[None, None], 
                                              k=5, s=0.3, 
                                              ext=0, check_finite=False)
    
    pol_flux_interped = np.linspace(0.0,2.0,501)
    ne_interped = interp_ne(pol_flux_interped)
    
    # plt.figure()
    # plt.plot(pol_flux_interped,ne_interped)
    
    ## Setting small values of ne to zero
    idx_last = len(ne_interped)
    for ii, ne in enumerate(ne_interped):
        if ne > ne_interped[0:10].max()*10**(-3):     
            continue
        else: 
            idx_last = ii
            break
    ne_trimmed = ne_interped[0:idx_last]
    polflux_trimmed = pol_flux_interped[0:idx_last]
    
    pol_flux_enter = fsolve(interp_ne, polflux_trimmed[-1])
    ne_output = np.append(ne_trimmed,0)
    polflux_output = np.append(polflux_trimmed,pol_flux_enter)    
    return polflux_output, ne_output


input_files_path ='C:\\Dropbox\\VHChen2023\\Data - Equilibrium\\'

suffix = '_188842_2500ms'
ne_filename = input_files_path + 'ne' +suffix+ '.dat'

ne_data = np.fromfile(ne_filename,dtype=float, sep='   ') # electron density as a function of poloidal flux label

ne_data_length = int(ne_data[0])
ne_data_density_array = ne_data[2::2] # in units of 10.0**19 m-3
ne_data_radialcoord_array = ne_data[1::2]   
ne_data_flux_array = ne_data_radialcoord_array**2
# ne_data_flux_array = ne_data[1::2] # My new IDL file outputs the flux density directly, instead of radialcoord

truncate_core = False

# Loading from DIII-D
ne_data_density_array = ne_data_density_array / 10**19
##


plt.figure()
plt.scatter(ne_data_flux_array,ne_data_density_array,color='black')
# plt.ylim([0, 6])
# plt.xlim([0, 1.5])

pol_flux_input = ne_data_flux_array
ne_input = ne_data_density_array
ne_err_input = np.zeros_like(ne_input)

## Pad
    ## Pads negative values of density in the edge
    ## This is so that the spline will reach zero at some point
    ## Negative values removed later
pol_flux_padding = np.linspace(pol_flux_input.max(),pol_flux_input.max()+0.5,51)
pol_flux_padded = np.append(pol_flux_input,pol_flux_padding)
ne_padded = np.append(ne_input,-0.1*np.ones_like(pol_flux_padding))
ne_err_padded = np.append(ne_err_input,np.zeros_like(pol_flux_padding))

if truncate_core:
    ## Truncate
        ## Sometimes, there is a spurious density peak near the core
        ## This bit removes the points in the core
    pol_flux_start = 0.2
    idx_remove = pol_flux_padded < pol_flux_start
    pol_flux_truncated = pol_flux_padded[~idx_remove]
    ne_truncated = ne_padded[~idx_remove]
    ne_err_truncated = ne_err_padded[~idx_remove]
    
    ## Pad (again)
        ## Pads density near the core
        ## This is so that the spline will have a flat density at the core
    ne_pad = 1.32
    pol_flux_padding = np.linspace(0,pol_flux_start,51)
    pol_flux_padded = np.append(pol_flux_padding,pol_flux_truncated)
    ne_padded = np.append(ne_pad*np.ones_like(pol_flux_padding),ne_truncated)
    ne_err_padded = np.append(np.zeros_like(pol_flux_padding),ne_err_truncated)


pol_flux_to_use = pol_flux_padded
ne_to_use = ne_padded
ne_err_to_use = ne_err_padded    


    ## Interp and smooth with splines

polflux_output, ne_smoothed = interp_smooth_Thomson(pol_flux_to_use, ne_to_use, ne_err_to_use)
# polflux_output, ne_output = smooth_Thomson(pol_flux_to_use, ne_to_use)

## Filter function
filter_steepness = 10.0
filter_centre = 1.3
filter_amp = 1.0
filter_fun = 0.5*filter_amp*(1-np.tanh(filter_steepness * (polflux_output - filter_centre))) + (1-filter_amp)
ne_output = ne_smoothed * filter_fun


# plt.errorbar(pol_flux_outboard,ne_outboard,ne_err_outboard,linestyle='',marker='.',mfc='none',label='outboard')
plt.plot(polflux_output,ne_smoothed,'k',linestyle='-',marker='',mfc='none',label='all')
plt.plot(polflux_output,filter_fun,'gray',linestyle='-',marker='',mfc='none',label='all')
plt.plot(polflux_output,ne_output,'r',linestyle='-',marker='',mfc='none',label='all')

# plt.plot(pol_flux_interped,ne_interped,'k',linestyle='-',marker='',mfc='none',label='outboard')

# plt.plot(pol_flux_plot,F_full(pol_flux_plot,*F_full_guess),'c',label='fit full')
# plt.plot(pol_flux_plot,F_ped(pol_flux_plot,*F_ped_guess),'m',label='fit ped')
# plt.plot(pol_flux_plot, F_full(pol_flux_plot,*popt))
# plt.plot(pol_flux_plot, popt[2] * np.tanh(popt[3] * (pol_flux_plot - popt[4])) )
# plt.errorbar(pol_flux_noNaN, ne_noNaN, yerr=ne_err_noNaN, fmt='-')
plt.xlim(-0.1,1.6)
# plt.ylim(0,1.1*ne_noNaN.max())
plt.ylim(-0.1,ne_output.max()+0.5)
plt.ylabel(r'$n_e / 10^{19} m^{-3}$') # x-direction
plt.xlabel(r'$\psi_p$')
plt.title('Density profile')
plt.legend(loc='upper right',edgecolor='k',facecolor=None)
plt.savefig('ne.jpg',dpi=300)
    
plt.savefig('Thomson' + '.jpg', dpi=300, format='jpg', bbox_inches='tight')

ne_data_file = open('ne_processed.dat','w')  
ne_data_file.write(str(len(polflux_output)) + '\n') 
for ii in range(0, len(polflux_output)):
    ## Saving radial coordinate rather than poloidal flux
    ne_data_file.write('{:.8e}   {:.8e} \n'.format(np.sqrt(polflux_output[ii]),ne_output[ii]))        
ne_data_file.close() 
