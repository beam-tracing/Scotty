# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:52:51 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt




loadfile = np.load('data_output1.npz')
q_R_array1 = loadfile['q_R_array']
q_zeta_array1 = loadfile['q_zeta_array']
q_Z_array1 = loadfile['q_Z_array']
K_R_array1 = loadfile['K_R_array']
K_zeta_initial1 = loadfile['K_zeta_initial']
K_Z_array1 = loadfile['K_Z_array']
dH_dKR_output1 = loadfile['dH_dKR_output']
dH_dKzeta_output1 = loadfile['dH_dKzeta_output']
dH_dKZ_output1 = loadfile['dH_dKZ_output']
dH_dR_output1 = loadfile['dH_dR_output']
dH_dZ_output1 = loadfile['dH_dZ_output']
distance_from_launch_to_entry1 = loadfile['distance_from_launch_to_entry']
loadfile.close()

loadfile = np.load('analysis_output1.npz')
distance_along_line1 = loadfile['distance_along_line']
loadfile.close()

loadfile = np.load('data_output2.npz')
q_R_array2 = loadfile['q_R_array']
q_zeta_array2 = loadfile['q_zeta_array']
q_Z_array2 = loadfile['q_Z_array']
K_R_array2 = loadfile['K_R_array']
K_zeta_initial2 = loadfile['K_zeta_initial']
K_Z_array2 = loadfile['K_Z_array']
dH_dKR_output2 = loadfile['dH_dKR_output']
dH_dKzeta_output2 = loadfile['dH_dKzeta_output']
dH_dKZ_output2 = loadfile['dH_dKZ_output']
dH_dR_output2 = loadfile['dH_dR_output']
dH_dZ_output2 = loadfile['dH_dZ_output']
distance_from_launch_to_entry2 = loadfile['distance_from_launch_to_entry']
loadfile.close()

loadfile = np.load('analysis_output2.npz')
distance_along_line2 = loadfile['distance_along_line']
in_index = loadfile['in_index']
out_index = loadfile['out_index']
loadfile.close()

loadfile = np.load('data_output3.npz')
q_R_array3 = loadfile['q_R_array']
q_zeta_array3 = loadfile['q_zeta_array']
q_Z_array3 = loadfile['q_Z_array']
K_R_array3 = loadfile['K_R_array']
K_zeta_initial3 = loadfile['K_zeta_initial']
K_Z_array3 = loadfile['K_Z_array']
dH_dKR_output3 = loadfile['dH_dKR_output']
dH_dKzeta_output3 = loadfile['dH_dKzeta_output']
dH_dKZ_output3 = loadfile['dH_dKZ_output']
dH_dR_output3 = loadfile['dH_dR_output']
dH_dZ_output3 = loadfile['dH_dZ_output']
distance_from_launch_to_entry3 = loadfile['distance_from_launch_to_entry']
loadfile.close()

loadfile = np.load('analysis_output3.npz')
distance_along_line3 = loadfile['distance_along_line']
loadfile.close()

plt.figure()
plt.plot(q_R_array1,q_Z_array1,'r')
plt.plot(q_R_array2,q_Z_array2,'g')
plt.plot(q_R_array3,q_Z_array3,'b')

plt.figure()
plt.polar(q_zeta_array1,q_R_array1,'r')
plt.polar(q_zeta_array2,q_R_array2,'g')
plt.polar(q_zeta_array3,q_R_array3,'b')
plt.xlim([0, 3.141592654/4])

plt.figure()
plt.subplot(2,3,1)
plt.plot(distance_along_line1,q_R_array1,'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,q_R_array2,'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,q_R_array3,'b')
plt.axvline(distance_along_line1[in_index],color='k')
plt.axvline(distance_along_line1[out_index],color='k')
plt.title('R')

plt.subplot(2,3,2)
plt.plot(distance_along_line1,q_zeta_array1,'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,q_zeta_array2,'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,q_zeta_array3,'b')
plt.axvline(distance_along_line1[in_index],color='k')
plt.axvline(distance_along_line1[out_index],color='k')
plt.title('zeta')

plt.subplot(2,3,3)
plt.plot(distance_along_line1,q_Z_array1,'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,q_Z_array2,'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,q_Z_array3,'b')
plt.axvline(distance_along_line1[in_index],color='k')
plt.axvline(distance_along_line1[out_index],color='k')
plt.title('Z')

plt.subplot(2,3,4)
plt.plot(distance_along_line1,K_R_array1,'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,K_R_array2,'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,K_R_array3,'b')
plt.axvline(distance_along_line1[in_index],color='k')
plt.axvline(distance_along_line1[out_index],color='k')
plt.title('K_R')

plt.subplot(2,3,5)


plt.subplot(2,3,6)
plt.plot(distance_along_line1,K_Z_array1,'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,K_Z_array2,'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,K_Z_array3,'b')
plt.axvline(distance_along_line1[in_index],color='k')
plt.axvline(distance_along_line1[out_index],color='k')
plt.title('K_Z')


plt.figure()
plt.subplot(2,3,1)
plt.plot(distance_along_line1,dH_dR_output1,'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,dH_dR_output2,'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,dH_dR_output3,'b')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('dH_dR')

plt.subplot(2,3,2)


plt.subplot(2,3,3)
plt.plot(distance_along_line1,dH_dZ_output1,'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,dH_dZ_output2,'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,dH_dZ_output3,'b')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('dH_dZ')

plt.subplot(2,3,4)
plt.plot(distance_along_line1,dH_dKR_output1,'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,dH_dKR_output2,'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,dH_dKR_output3,'b')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('dH_dKR')

plt.subplot(2,3,5)
plt.plot(distance_along_line1,dH_dKzeta_output1,'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,dH_dKzeta_output2,'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,dH_dKzeta_output3,'b')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('dH_dKzeta')

plt.subplot(2,3,6)
plt.plot(distance_along_line1,dH_dKZ_output1,'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,dH_dKZ_output2,'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,dH_dKZ_output3,'b')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('dH_dKZ')











#numberOfDataPoints = len(distance_along_line1)
#point_spacing1_cyl = np.zeros(numberOfDataPoints)
#point_spacing2_cyl = np.zeros(numberOfDataPoints)
#for ii in range(1,numberOfDataPoints):
#    point_spacing1_cyl[ii] = np.sqrt(q_R_array1[ii]**2 + q_R_array1[ii-1]**2 
#            - 2 * q_R_array1[ii] * q_R_array1[ii-1] * np.cos(q_zeta_array1[ii] - q_zeta_array1[ii-1])
#            + (q_Z_array1[ii] - q_Z_array1[ii-1])**2
#            )
#    point_spacing2_cyl[ii] = np.sqrt(q_R_array2[ii]**2 + q_R_array2[ii-1]**2 
#            - 2 * q_R_array2[ii] * q_R_array2[ii-1] * np.cos(q_zeta_array2[ii] - q_zeta_array2[ii-1])
#            + (q_Z_array2[ii] - q_Z_array2[ii-1])**2
#            )    
#
#distance_along_line1_cyl =  np.cumsum(point_spacing1_cyl)
#distance_along_line1_cyl = np.append(0,distance_along_line1_cyl)
#
#distance_along_line2_cyl =  np.cumsum(point_spacing2_cyl)
#distance_along_line2_cyl = np.append(0,distance_along_line2_cyl)
#
#
#plt.figure()
#plt.plot(distance_along_line1)
#plt.plot(distance_along_line1_cyl)
#    
#
#plt.figure()
#plt.plot(distance_along_line2)
#plt.plot(distance_along_line2_cyl)
#
#
#    
#    
#    
    
    
    
    
    
    