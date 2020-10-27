# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:37:48 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import interpolate as interpolate


loadfile = np.load('analysis_output.npz')
localisation_piece = loadfile['localisation_piece']
cutoff_index = loadfile['cutoff_index']
RZ_distance_along_line = loadfile['RZ_distance_along_line']
distance_along_line = loadfile['distance_along_line']
k_perp_1_backscattered = loadfile['k_perp_1_backscattered']
Psi_xx_output = loadfile['Psi_xx_output']
Psi_xy_output = loadfile['Psi_xy_output']
Psi_yy_output = loadfile['Psi_yy_output']
M_xx_output = loadfile['M_xx_output']
M_xy_output = loadfile['M_xy_output']
M_yy_output = loadfile['M_yy_output']
loadfile.close()
#
#start_index = 200
#end_index = 1000
#
## Ray: Localisation vs distance
#moving_int_localisation_piece = np.zeros((end_index-start_index))
#window = 100
#for ii in range(0,(end_index-start_index)):
#    moving_int_localisation_piece[ii] = simps(abs(localisation_piece[start_index+ii-window:start_index+ii+window])) / (distance_along_line[start_index+ii+window] - distance_along_line[start_index+ii-window])
#
#
#plt.figure(figsize=(11.0, 5.0))
#plt.subplot(1,3,1)
#plt.plot(distance_along_line[start_index:end_index],localisation_piece[start_index:end_index],'.')
#plt.axvline(distance_along_line[cutoff_index],color='k')
#plt.ylim([0, 0.5])
#plt.xlabel('l / m') # x-direction
##plt.title(r'$\frac{g^2_{ant}}{|g \frac{\rmd K}{\rmd \tau} |}$')
#plt.title('Ray')
#
#
#plt.subplot(1,3,2)
#plt.plot(distance_along_line[start_index:end_index],moving_int_localisation_piece/moving_int_localisation_piece.max(),'.')
#plt.axvline(distance_along_line[cutoff_index],color='k')
#plt.ylabel('Normalised to peak')
#plt.xlabel('l / m') # x-direction
#
##plt.title('Moving average of ' + r'$\frac{g^2_{ant}}{|g \frac{\rmd K}{\rmd \tau} |}$')
#plt.title('Ray (avr)')
#
#
#
## Beam: Localisation vs distance
#numberOfDataPoints = len(M_xx_output)             
#M_w = np.zeros([numberOfDataPoints,2,2],dtype='complex128')
#M_w[:,0,0] = M_xx_output
#M_w[:,1,1] = M_yy_output
#M_w[:,1,0] = M_xy_output
#M_w[:,0,1] = M_w[:,1,0]
#
#Psi_w = np.zeros([numberOfDataPoints,2,2],dtype='complex128')
#Psi_w[:,0,0] = Psi_xx_output
#Psi_w[:,1,1] = Psi_yy_output
#Psi_w[:,1,0] = Psi_xy_output
#Psi_w[:,0,1] = Psi_w[:,1,0]
#         
#beam_localisation_piece = np.linalg.det(np.imag(Psi_w)) / abs(np.linalg.det(M_w))
#
#
#
#plt.subplot(1,3,3)
#plt.plot(distance_along_line[start_index:end_index],beam_localisation_piece[start_index:end_index],'.')
#plt.axvline(distance_along_line[cutoff_index],color='k')
#plt.title('Beam')
#plt.xlabel('l / m') # x-direction
#plt.tight_layout()
#plt.savefig('Localisation', bbox_inches='tight')
##plt.savefig('Localisation.eps', format='eps', bbox_inches='tight')
#plt.savefig('Localisation.jpg', format='jpg', bbox_inches='tight')
#
#
#
#
#
#
#interp_localisation_piece = interpolate.interp1d(distance_along_line[start_index:end_index], localisation_piece[start_index:end_index],
#                                             kind='linear', axis=-1, copy=True, bounds_error=False,
#                                             fill_value=0, assume_sorted=False) # localisation piece is 0 outside the LCFS
#
#
#
#
#
#numberOfPoints = 50
#distance_array=np.linspace(0.0,1.0,10001)
#delta_distance=distance_array[1]*numberOfPoints
#
#localisation_piece_interp = interp_localisation_piece(distance_array)
#moving_int_localisation_piece_interp = np.zeros(10001)
#moving_int_localisation_piece_interp2 = np.zeros(10001)
#
#for ii in range(numberOfPoints*5,10001-numberOfPoints*5):
#    moving_int_localisation_piece_interp[ii] = simps(abs( localisation_piece_interp[ii-numberOfPoints:ii+numberOfPoints] )) / (2*delta_distance)
#    moving_int_localisation_piece_interp2[ii] = simps(abs( localisation_piece_interp[ii-numberOfPoints*5:ii+numberOfPoints*5] )) / (2*delta_distance*5)
#
#plt.figure(figsize=(11.0, 5.0))
#plt.subplot(1,2,1)
#plt.plot(distance_array,localisation_piece_interp,'.')
#plt.axvline(distance_along_line[cutoff_index],color='k')
#plt.ylim([0, 0.5])
#plt.xlim([0.2, 0.8])
#plt.xlabel('l / m') # x-direction
##plt.title(r'$\frac{g^2_{ant}}{|g \frac{\rmd K}{\rmd \tau} |}$')
#plt.title('Ray')
#
#
#plt.subplot(1,2,2)
#plt.plot(distance_array,moving_int_localisation_piece_interp/moving_int_localisation_piece_interp.max(),'.',label= 'width = ' + str(2*delta_distance) +'m')
#plt.plot(distance_array,moving_int_localisation_piece_interp2/moving_int_localisation_piece_interp2.max(),'.',label='width = ' + str(2*delta_distance*5) +'m')
#plt.xlim([0.2, 0.8])
#plt.axvline(distance_along_line[cutoff_index],color='k')
#plt.ylabel('Normalised to peak')
#plt.xlabel('l / m') # x-direction
#plt.legend()
#
##plt.title('Moving average of ' + r'$\frac{g^2_{ant}}{|g \frac{\rmd K}{\rmd \tau} |}$')
#plt.title('Ray (avr)')
#plt.savefig('Localisation2.jpg', format='jpg', bbox_inches='tight')




tau_s_index = np.argmin(abs(k_perp_1_backscattered))
tau_vac_index = np.argmax(abs(k_perp_1_backscattered))

k_perp_1_backscattered_a = k_perp_1_backscattered[0:tau_s_index]
k_perp_1_backscattered_b = k_perp_1_backscattered[tau_s_index::]
localisation_piece_a = localisation_piece[0:tau_s_index]
localisation_piece_b = localisation_piece[tau_s_index::]

interp_localisation_piece_a = interpolate.interp1d(k_perp_1_backscattered_a, localisation_piece_a,
                                             kind='linear', axis=-1, copy=True, bounds_error=False,
                                             fill_value=0, assume_sorted=False) 
interp_localisation_piece_b = interpolate.interp1d(k_perp_1_backscattered_b, localisation_piece_b,
                                             kind='linear', axis=-1, copy=True, bounds_error=False,
                                             fill_value=0, assume_sorted=False) 

numberOfPoints = 100
arraySize = 20001
k_perp_1_array = np.linspace(1.5*k_perp_1_backscattered[tau_vac_index],0, arraySize )
delta_kperp1_1 = (k_perp_1_array[1]-k_perp_1_array[0])*numberOfPoints

moving_int_localisation_piece_interp_a = np.zeros(arraySize)
moving_int_localisation_piece_interp_b = np.zeros(arraySize)
localisation_piece_interp_a = interp_localisation_piece_a(k_perp_1_array)
localisation_piece_interp_b = interp_localisation_piece_a(k_perp_1_array)

for ii in range(numberOfPoints,arraySize-numberOfPoints):
    moving_int_localisation_piece_interp_a[ii] = simps(abs( localisation_piece_interp_a[ii-numberOfPoints:ii+numberOfPoints] )) / (2*delta_kperp1_1)
    moving_int_localisation_piece_interp_b[ii] = simps(abs( localisation_piece_interp_b[ii-numberOfPoints:ii+numberOfPoints] )) / (2*delta_kperp1_1)

moving_int_localisation_piece_interp_1 = moving_int_localisation_piece_interp_a+moving_int_localisation_piece_interp_b


numberOfPoints = 500
moving_int_localisation_piece_interp_a = np.zeros(arraySize)
moving_int_localisation_piece_interp_b = np.zeros(arraySize)
delta_kperp1_2 = (k_perp_1_array[1]-k_perp_1_array[0])*numberOfPoints
for ii in range(numberOfPoints,arraySize-numberOfPoints):
    moving_int_localisation_piece_interp_a[ii] = simps(abs( localisation_piece_interp_a[ii-numberOfPoints:ii+numberOfPoints] )) / (2*delta_kperp1_2)
    moving_int_localisation_piece_interp_b[ii] = simps(abs( localisation_piece_interp_b[ii-numberOfPoints:ii+numberOfPoints] )) / (2*delta_kperp1_2)
    
moving_int_localisation_piece_interp_2 = moving_int_localisation_piece_interp_a+moving_int_localisation_piece_interp_b

    
plt.figure(figsize=(11.0, 5.0))
plt.subplot(1,2,1)
plt.plot(k_perp_1_backscattered_a,localisation_piece_a,'.')
plt.plot(k_perp_1_backscattered_b,localisation_piece_b,'.')
plt.ylim([0, 0.1])
plt.xlim([k_perp_1_backscattered[tau_vac_index], k_perp_1_backscattered[tau_s_index]])
plt.xlabel(r'$ k_{\perp,1} / m^{-1} $') # x-direction
plt.title(r'$ g^2_{ant} / |g \frac{\rmd K}{\rmd \tau} |$')


plt.subplot(1,2,2)
plt.plot(k_perp_1_array,moving_int_localisation_piece_interp_1,'g')
plt.plot(k_perp_1_array,moving_int_localisation_piece_interp_2,'b')

plt.plot([-2000, -2000+2*delta_kperp1_1],[0.7, 0.7],'g')
plt.plot([-2000, -2000+2*delta_kperp1_2],[0.65, 0.65],'b')
plt.text(-2275, 0.665, 'window', fontsize=12)
plt.xlim([k_perp_1_backscattered[tau_vac_index], k_perp_1_backscattered[tau_s_index]])
plt.xlabel(r'$ k_{\perp,1} / m^{-1} $') # x-direction
#plt.ylabel('Normalised to peak')
#plt.xlabel('l / m') # x-direction


#plt.title('Moving average of ' + r'$\frac{g^2_{ant}}{|g \frac{\rmd K}{\rmd \tau} |}$')
plt.title('Ray (avr)')
plt.savefig('Localisation3.jpg', format='jpg', bbox_inches='tight')


#
##Localisation vs k
##k_perp_1_backscattered_truncated = k_perp_1_backscattered[start_index:end_index]
##k_perp_1_min = k_perp_1_backscattered_truncated.min()
##k_perp_1_max = k_perp_1_backscattered_truncated.max()
##k_perp_1_array = np.linspace(k_perp_1_min,k_perp_1_max,1000)
#
#localisation_piece_a = localisation_piece[start_index:cutoff_index]
#localisation_piece_b = localisation_piece[cutoff_index:end_index]
#k_perp_1_a = k_perp_1_backscattered[start_index:cutoff_index]
#
#moving_int_localisation_piece_a = np.zeros(len(localisation_piece_a))
#for ii in range(0,len(localisation_piece_a)):
#    moving_int_localisation_piece_a[ii] = simps(k_perp_1_backscattered[start_index+ii-2:start_index+ii+2]) / (k_perp_1_backscattered[start_index+ii+2] - k_perp_1_backscattered[start_index+ii-2])
#    
#    
#plt.figure()
#plt.plot(k_perp_1_a,moving_int_localisation_piece_a)    
#    
#plt.figure()
#plt.plot(k_perp_1_backscattered[start_index:cutoff_index],localisation_piece[start_index:cutoff_index],'g.')
#plt.plot(k_perp_1_backscattered[cutoff_index:end_index],localisation_piece[cutoff_index:end_index],'b.')
#
#plt.title(r'$\frac{g^2_{ant}}{|g \frac{\rmd K}{\rmd \tau} |}$')
