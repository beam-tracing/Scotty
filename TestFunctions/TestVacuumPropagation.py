# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:52:51 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt




loadfile = np.load('data_output.npz')
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
grad_grad_H_output1 = loadfile['grad_grad_H_output']
gradK_grad_H_output1 = loadfile['gradK_grad_H_output']
gradK_gradK_H_output1 = loadfile['gradK_gradK_H_output']
loadfile.close()

loadfile = np.load('analysis_output1.npz')
distance_along_line1 = loadfile['distance_along_line']
in_index = loadfile['in_index']
out_index = loadfile['out_index']
Psi_xx_output1 = loadfile['Psi_xx_output']
Psi_xy_output1 = loadfile['Psi_xy_output']
Psi_yy_output1 = loadfile['Psi_yy_output']
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
grad_grad_H_output2 = loadfile['grad_grad_H_output']
gradK_grad_H_output2 = loadfile['gradK_grad_H_output']
gradK_gradK_H_output2 = loadfile['gradK_gradK_H_output']
loadfile.close()

loadfile = np.load('analysis_output2.npz')
distance_along_line2 = loadfile['distance_along_line']
Psi_xx_output2 = loadfile['Psi_xx_output']
Psi_xy_output2 = loadfile['Psi_xy_output']
Psi_yy_output2 = loadfile['Psi_yy_output']
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
grad_grad_H_output3 = loadfile['grad_grad_H_output']
gradK_grad_H_output3 = loadfile['gradK_grad_H_output']
gradK_gradK_H_output3 = loadfile['gradK_gradK_H_output']
loadfile.close()

loadfile = np.load('analysis_output3.npz')
distance_along_line3 = loadfile['distance_along_line']
Psi_xx_output3 = loadfile['Psi_xx_output']
Psi_xy_output3 = loadfile['Psi_xy_output']
Psi_yy_output3 = loadfile['Psi_yy_output']
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















plt.figure()
plt.subplot(2,2,1)
plt.plot(distance_along_line1,np.real(Psi_xx_output1),'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,np.real(Psi_xx_output2),'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,np.real(Psi_xx_output3),'b')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('Re [Psi_xx]')

plt.subplot(2,2,2)
plt.plot(distance_along_line1,np.real(Psi_xy_output1),'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,np.real(Psi_xy_output2),'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,np.real(Psi_xy_output3),'b')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('Re [Psi_xy]')

plt.subplot(2,2,3)


plt.subplot(2,2,4)
plt.plot(distance_along_line1,np.real(Psi_yy_output1),'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,np.real(Psi_yy_output2),'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,np.real(Psi_yy_output3),'b')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('Re [Psi_yy]')




plt.figure()
plt.subplot(2,2,1)
plt.plot(distance_along_line1,np.imag(Psi_xx_output1),'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,np.imag(Psi_xx_output2),'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,np.imag(Psi_xx_output3),'b')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('Im [Psi_xx]')

plt.subplot(2,2,2)
plt.plot(distance_along_line1,np.imag(Psi_xy_output1),'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,np.imag(Psi_xy_output2),'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,np.imag(Psi_xy_output3),'b')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('Im [Psi_xy]')

plt.subplot(2,2,3)


plt.subplot(2,2,4)
plt.plot(distance_along_line1,np.imag(Psi_yy_output1),'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,np.imag(Psi_yy_output2),'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,np.imag(Psi_yy_output3),'b')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('Im [Psi_yy]')



plt.figure()
plt.subplot(3,3,1)
plt.plot(distance_along_line1,grad_grad_H_output1[:,0,0],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,grad_grad_H_output2[:,0,0],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,grad_grad_H_output3[:,0,0],'b')

plt.subplot(3,3,2)

plt.subplot(3,3,3)
plt.plot(distance_along_line1,grad_grad_H_output1[:,0,2],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,grad_grad_H_output2[:,0,2],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,grad_grad_H_output3[:,0,2],'b')

plt.subplot(3,3,4)

plt.subplot(3,3,5)

plt.subplot(3,3,6)

plt.subplot(3,3,7)
#plt.plot(distance_along_line1,grad_grad_H_output1[:,2,0],'r')
#plt.plot(distance_along_line2+distance_from_launch_to_entry2,grad_grad_H_output2[:,2,0],'g')
#plt.plot(distance_along_line3+distance_from_launch_to_entry3,grad_grad_H_output3[:,2,0],'b')

plt.subplot(3,3,8)

plt.subplot(3,3,9)
plt.plot(distance_along_line1,grad_grad_H_output1[:,2,2],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,grad_grad_H_output2[:,2,2],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,grad_grad_H_output3[:,2,2],'b')





plt.figure()
plt.subplot(3,3,1)
plt.plot(distance_along_line1,gradK_grad_H_output1[:,0,0],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_grad_H_output2[:,0,0],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_grad_H_output3[:,0,0],'b')

plt.subplot(3,3,2)

plt.subplot(3,3,3)
plt.plot(distance_along_line1,gradK_grad_H_output1[:,0,2],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_grad_H_output2[:,0,2],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_grad_H_output3[:,0,2],'b')

plt.subplot(3,3,4)
plt.plot(distance_along_line1,gradK_grad_H_output1[:,1,0],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_grad_H_output2[:,1,0],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_grad_H_output3[:,1,0],'b')

plt.subplot(3,3,5)

plt.subplot(3,3,6)
plt.plot(distance_along_line1,gradK_grad_H_output1[:,1,2],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_grad_H_output2[:,1,2],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_grad_H_output3[:,1,2],'b')

plt.subplot(3,3,7)
plt.plot(distance_along_line1,gradK_grad_H_output1[:,2,0],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_grad_H_output2[:,2,0],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_grad_H_output3[:,2,0],'b')

plt.subplot(3,3,8)

plt.subplot(3,3,9)
plt.plot(distance_along_line1,gradK_grad_H_output1[:,2,2],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_grad_H_output2[:,2,2],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_grad_H_output3[:,2,2],'b')



plt.figure()
plt.subplot(3,3,1)
plt.plot(distance_along_line1,gradK_gradK_H_output1[:,0,0],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_gradK_H_output2[:,0,0],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_gradK_H_output3[:,0,0],'b')

plt.subplot(3,3,2)
plt.plot(distance_along_line1,gradK_gradK_H_output1[:,0,1],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_gradK_H_output2[:,0,1],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_gradK_H_output3[:,0,1],'b')

plt.subplot(3,3,3)
plt.plot(distance_along_line1,gradK_gradK_H_output1[:,0,2],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_gradK_H_output2[:,0,2],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_gradK_H_output3[:,0,2],'b')

plt.subplot(3,3,4)
#plt.plot(distance_along_line1,gradK_gradK_H_output1[:,1,0],'r')
#plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_gradK_H_output2[:,1,0],'g')
#plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_gradK_H_output3[:,1,0],'b')

plt.subplot(3,3,5)
plt.plot(distance_along_line1,gradK_gradK_H_output1[:,1,1],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_gradK_H_output2[:,1,1],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_gradK_H_output3[:,1,1],'b')

plt.subplot(3,3,6)
plt.plot(distance_along_line1,gradK_gradK_H_output1[:,1,2],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_gradK_H_output2[:,1,2],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_gradK_H_output3[:,1,2],'b')

plt.subplot(3,3,7)
#plt.plot(distance_along_line1,gradK_gradK_H_output1[:,2,0],'r')
#plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_gradK_H_output2[:,2,0],'g')
#plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_gradK_H_output3[:,2,0],'b')

plt.subplot(3,3,8)
#plt.plot(distance_along_line1,gradK_gradK_H_output1[:,2,1],'r')
#plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_gradK_H_output2[:,2,1],'g')
#plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_gradK_H_output3[:,2,1],'b')

plt.subplot(3,3,9)
plt.plot(distance_along_line1,gradK_gradK_H_output1[:,2,2],'r')
plt.plot(distance_along_line2+distance_from_launch_to_entry2,gradK_gradK_H_output2[:,2,2],'g')
plt.plot(distance_along_line3+distance_from_launch_to_entry3,gradK_gradK_H_output3[:,2,2],'b')




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
##    
#    
#    
#for ii in range(0,numberOfDataPoints):
#    find_widths_and_curvatures(Psi_xx, Psi_xy, Psi_yy,K_magnitude)
#    
#    
    