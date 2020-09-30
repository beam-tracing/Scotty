# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:37:48 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


loadfile = np.load('analysis_output.npz')
localisation_piece = loadfile['localisation_piece']
cutoff_index = loadfile['cutoff_index']
RZ_distance_along_line = loadfile['RZ_distance_along_line']
k_perp_1_backscattered = loadfile['k_perp_1_backscattered']
Psi_xx_output = loadfile['Psi_xx_output']
Psi_xy_output = loadfile['Psi_xy_output']
Psi_yy_output = loadfile['Psi_yy_output']
M_xx_output = loadfile['M_xx_output']
M_xy_output = loadfile['M_xy_output']
M_yy_output = loadfile['M_yy_output']
loadfile.close()

start_index = 50
end_index = 600

# Ray: Localisation vs distance
moving_int_localisation_piece = np.zeros((end_index-start_index))
window = 2
for ii in range(0,(end_index-start_index)):
    moving_int_localisation_piece[ii] = simps(abs(k_perp_1_backscattered[start_index+ii-window:start_index+ii+window])) / (RZ_distance_along_line[start_index+ii+window] - RZ_distance_along_line[start_index+ii-window])


plt.figure(figsize=(11.0, 5.0))
plt.subplot(1,3,1)
plt.plot(RZ_distance_along_line[start_index:end_index],localisation_piece[start_index:end_index],'.')
plt.axvline(RZ_distance_along_line[cutoff_index],color='k')
plt.ylim([0, 0.1])
plt.xlabel('l / m') # x-direction
#plt.title(r'$\frac{g^2_{ant}}{|g \frac{\rmd K}{\rmd \tau} |}$')
plt.title('Ray')


plt.subplot(1,3,2)
plt.plot(RZ_distance_along_line[start_index:end_index],moving_int_localisation_piece/moving_int_localisation_piece.max(),'.')
plt.axvline(RZ_distance_along_line[cutoff_index],color='k')
plt.ylabel('Normalised to peak')
plt.xlabel('l / m') # x-direction

#plt.title('Moving average of ' + r'$\frac{g^2_{ant}}{|g \frac{\rmd K}{\rmd \tau} |}$')
plt.title('Ray (avr)')



# Beam: Localisation vs distance
numberOfDataPoints = len(M_xx_output)             
M_w = np.zeros([numberOfDataPoints,2,2],dtype='complex128')
M_w[:,0,0] = M_xx_output
M_w[:,1,1] = M_yy_output
M_w[:,1,0] = M_xy_output
M_w[:,0,1] = M_w[:,1,0]

Psi_w = np.zeros([numberOfDataPoints,2,2],dtype='complex128')
Psi_w[:,0,0] = Psi_xx_output
Psi_w[:,1,1] = Psi_yy_output
Psi_w[:,1,0] = Psi_xy_output
Psi_w[:,0,1] = Psi_w[:,1,0]
         
beam_localisation_piece = np.linalg.det(np.imag(Psi_w)) / abs(np.linalg.det(M_w))



plt.subplot(1,3,3)
plt.plot(RZ_distance_along_line[start_index:end_index],beam_localisation_piece[start_index:end_index],'.')
plt.axvline(RZ_distance_along_line[cutoff_index],color='k')
plt.title('Beam')
plt.xlabel('l / m') # x-direction
plt.tight_layout()
plt.savefig('Localisation', bbox_inches='tight')
plt.savefig('Localisation.eps', format='eps', bbox_inches='tight')



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
