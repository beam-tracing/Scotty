# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

Functions for Scotty (excluding functions for finding derivatives of H).

@author: chenv
Valerian Hongjie Hall-Chen
valerian@hall-chen.com

Run in Python 3,  does not work in Python 2
"""

import numpy as np
from scipy import constants as constants
from scipy import interpolate as interpolate
from scipy import optimize as optimize


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


def find_nearest(array,  value): #returns the index
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)


def contract_special(arg_a,arg_b):
    """
    Takes 
    Either:
        matrix of TxMxN and a vector of TxN  or TxM
        For each T, contract the matrix with the vector
    Or:
        two vectors of size TxN
    For each T, contract the indices N 
    Covers the case that matmul and dot don't do very elegantly
    Avoids having to use a for loop to iterate over T
    """
    if (np.ndim(arg_a) == 3 and np.ndim(arg_b) == 2): # arg_a is the matrix and arg_b is the vector
        matrix = arg_a
        vector = arg_b
        intermediate_result = np.tensordot(matrix,vector, ((2), (1)))
        result = np.diagonal(intermediate_result, offset=0, axis1=0, axis2=2).transpose()
    elif (np.ndim(arg_a) == 2 and np.ndim(arg_b) == 3): # arg_a is the vector and arg_b is the matrix
        vector = arg_a
        matrix = arg_b
        intermediate_result = np.tensordot(matrix,vector, ((1), (1)))
        result = np.diagonal(intermediate_result, offset=0, axis1=0, axis2=2).transpose()
    elif (np.ndim(arg_a) == 2 and np.ndim(arg_b) == 2): # arg_a is the vector and arg_b is a vector
        vector1 = arg_a
        vector2 = arg_b
        intermediate_result = np.tensordot(vector1,vector2, ((1), (1)))
        result = np.diagonal(intermediate_result, offset=0, axis1=0, axis2=1).transpose()
    else: 
        print('Error: Invalid dimensions')
    return result


def find_inverse_2D(matrix_2D):
    # Finds the inverse of a 2x2 matrix
    matrix_2D_inverse = np.zeros([2,2],dtype='complex128')
    determinant = matrix_2D[0,0]*matrix_2D[1,1] - matrix_2D[0,1]*matrix_2D[1,0]
    matrix_2D_inverse[0,0] =   matrix_2D[1,1] / determinant
    matrix_2D_inverse[1,1] =   matrix_2D[0,0] / determinant
    matrix_2D_inverse[0,1] = - matrix_2D[0,1] / determinant
    matrix_2D_inverse[1,0] = - matrix_2D[1,0] / determinant
    return matrix_2D_inverse


def find_x0(xs,ys,y0):
    """
    xs,ys are the x and y coordinates of a line on a plane
    Finds the value of x corresponding to a certain y0
    This implementation is silly but I am impatient and want to move on to other things quickly
    """

    index_guess = find_nearest(ys,y0)

    interp_y = interpolate.interp1d(xs, ys,
                                    kind='linear', axis=-1, copy=True, bounds_error=False,
                                    fill_value='extrapolate', assume_sorted=False) 

    if index_guess == 0:
        xs_fine = np.linspace(xs[0],xs[index_guess+1],101)
    elif index_guess == len(xs)-1:
        xs_fine = np.linspace(xs[index_guess-1],xs[-1],101)
    else:
        xs_fine = np.linspace(xs[index_guess-1],xs[index_guess+1],101)
    
    ys_fine = interp_y(xs_fine)

    index = find_nearest(ys_fine,y0)
    x0 = xs_fine[index]
    
    return x0

#----------------------------------
    
# Functions (Coordinate transformations)
def find_q_lab_Cartesian(q_lab):
    q_R    = q_lab[0]
    q_zeta = q_lab[1]
    q_Z    = q_lab[2]
    
    q_lab_Cartesian = np.zeros_like(q_lab)
    q_lab_Cartesian[0] = q_R * np.cos(q_zeta)
    q_lab_Cartesian[1] = q_R * np.sin(q_zeta)
    q_lab_Cartesian[2] = q_Z
    return q_lab_Cartesian

def find_q_lab(q_lab_Cartesian):
    q_X = q_lab_Cartesian[0]
    q_Y = q_lab_Cartesian[1]
    q_Z = q_lab_Cartesian[2]
    
    q_lab = np.zeros_like(q_lab_Cartesian)
    q_lab[0] = np.sqrt(q_X**2 + q_Y**2)
    q_lab[1] = np.arctan2(q_Y,q_X)
    q_lab[2] = q_Z    
    return q_lab
    
def find_K_lab_Cartesian(K_lab,q_lab):
    K_R    = K_lab[0]
    K_zeta = K_lab[1]
    K_Z    = K_lab[2]
    q_R    = q_lab[0]
    q_zeta = q_lab[1]
    
    K_lab_Cartesian = np.zeros(3)
    K_lab_Cartesian[0] = K_R*np.cos( q_zeta ) - K_zeta*np.sin( q_zeta ) / q_R #K_X
    K_lab_Cartesian[1] = K_R*np.sin( q_zeta ) + K_zeta*np.cos( q_zeta ) / q_R #K_Y
    K_lab_Cartesian[2] = K_Z
    return K_lab_Cartesian    
    
def find_K_lab(K_lab_Cartesian,q_lab_Cartesian):
    K_X = K_lab_Cartesian[0]
    K_Y = K_lab_Cartesian[1]
    K_Z = K_lab_Cartesian[2]
    
    [q_R,q_zeta,q_Z] = find_q_lab(q_lab_Cartesian)

    K_lab = np.zeros(3)
    K_lab[0] =   K_X*np.cos(q_zeta) + K_Y*np.sin(q_zeta) #K_R
    K_lab[1] = (-K_X*np.sin(q_zeta) + K_Y*np.cos(q_zeta))*q_R #K_zeta
    K_lab[2] =   K_Z
    return K_lab

def find_Psi_3D_lab(Psi_3D_lab_Cartesian, q_R, q_zeta, K_R, K_zeta):
    """
    Converts Psi_3D from Cartesian to cylindrical coordinates, both in the lab frame (not the beam frame)
    """
    cos_zeta = np.cos(q_zeta)
    sin_zeta = np.sin(q_zeta)
    
    Psi_XX = Psi_3D_lab_Cartesian[0][0]
    Psi_YY = Psi_3D_lab_Cartesian[1][1]
    Psi_ZZ = Psi_3D_lab_Cartesian[2][2]
    Psi_XY = Psi_3D_lab_Cartesian[0][1]
    Psi_XZ = Psi_3D_lab_Cartesian[0][2]
    Psi_YZ = Psi_3D_lab_Cartesian[1][2]

    Psi_3D_lab = np.zeros([3,3],dtype='complex128')
    
    Psi_3D_lab[0][0] = (
                            Psi_XX * cos_zeta**2 
                        + 2*Psi_XY * sin_zeta * cos_zeta 
                        +   Psi_YY * sin_zeta**2 
                       ) #Psi_RR
    Psi_3D_lab[1][1] = (
                           Psi_XX * sin_zeta**2
                       - 2*Psi_XY * sin_zeta * cos_zeta
                       +   Psi_YY * cos_zeta**2
                       )* q_R**2 - K_R    * q_R #Psi_zetazeta
    Psi_3D_lab[2][2] = Psi_ZZ #Psi_ZZ
    
    Psi_3D_lab[0][1] = (
                       -   Psi_XX * sin_zeta * cos_zeta
                       +   Psi_XY * (cos_zeta**2 - sin_zeta**2)
                       +   Psi_YY * sin_zeta * cos_zeta
                       )* q_R    + K_zeta / q_R #Psi_Rzeta
    Psi_3D_lab[1][0] = Psi_3D_lab[0][1]
    
    Psi_3D_lab[0][2] = (
                           Psi_XZ * cos_zeta
                       +   Psi_YZ * sin_zeta
                       ) #Psi_RZ
    Psi_3D_lab[2][0] = Psi_3D_lab[0][2]
    Psi_3D_lab[1][2] = (
                       -   Psi_XZ * sin_zeta
                       +   Psi_YZ * cos_zeta
                       )* q_R #Psi_zetaZ
    Psi_3D_lab[2][1] = Psi_3D_lab[1][2]
    return Psi_3D_lab
    
def find_Psi_3D_lab_Cartesian(Psi_3D_lab, q_R, q_zeta, K_R, K_zeta):
    """
    Converts Psi_3D from cylindrical to Cartesian coordinates, both in the lab frame (not the beam frame)
    The shape of Psi_3D_lab must be either [3,3] or [numberOfDataPoints,3,3]
    """
    if Psi_3D_lab.ndim == 2: # A single matrix of Psi
        Psi_RR       = Psi_3D_lab[0,0]
        Psi_zetazeta = Psi_3D_lab[1,1]
        Psi_ZZ       = Psi_3D_lab[2,2]
        Psi_Rzeta    = Psi_3D_lab[0,1]
        Psi_RZ       = Psi_3D_lab[0,2]
        Psi_zetaZ    = Psi_3D_lab[1,2]
        
        temp_matrix_for_Psi = np.zeros(np.shape(Psi_3D_lab),dtype='complex128')

        temp_matrix_for_Psi[0,0] = Psi_RR
        temp_matrix_for_Psi[0,1] = Psi_Rzeta   /q_R - K_zeta/q_R**2
        temp_matrix_for_Psi[0,2] = Psi_RZ
        temp_matrix_for_Psi[1,1] = Psi_zetazeta/q_R**2 + K_R/q_R
        temp_matrix_for_Psi[1,2] = Psi_zetaZ   /q_R
        temp_matrix_for_Psi[2,2] = Psi_ZZ
        temp_matrix_for_Psi[1,0] = temp_matrix_for_Psi[0,1]
        temp_matrix_for_Psi[2,0] = temp_matrix_for_Psi[0,2]
        temp_matrix_for_Psi[2,1] = temp_matrix_for_Psi[1,2]
    elif Psi_3D_lab.ndim == 3: # Matrices of Psi, residing in the first index
        Psi_RR       = Psi_3D_lab[:,0,0]
        Psi_zetazeta = Psi_3D_lab[:,1,1]
        Psi_ZZ       = Psi_3D_lab[:,2,2]
        Psi_Rzeta    = Psi_3D_lab[:,0,1]
        Psi_RZ       = Psi_3D_lab[:,0,2]
        Psi_zetaZ    = Psi_3D_lab[:,1,2]
        
        temp_matrix_for_Psi = np.zeros(np.shape(Psi_3D_lab),dtype='complex128')

        temp_matrix_for_Psi[:,0,0] = Psi_RR
        temp_matrix_for_Psi[:,0,1] = Psi_Rzeta   /q_R - K_zeta/q_R**2
        temp_matrix_for_Psi[:,0,2] = Psi_RZ
        temp_matrix_for_Psi[:,1,1] = Psi_zetazeta/q_R**2 + K_R/q_R
        temp_matrix_for_Psi[:,1,2] = Psi_zetaZ   /q_R
        temp_matrix_for_Psi[:,2,2] = Psi_ZZ
        temp_matrix_for_Psi[:,1,0] = temp_matrix_for_Psi[:,0,1]
        temp_matrix_for_Psi[:,2,0] = temp_matrix_for_Psi[:,0,2]
        temp_matrix_for_Psi[:,2,1] = temp_matrix_for_Psi[:,1,2]
    else:
        print('Error: Psi_3D_lab has an invalid number of dimensions')
        


    rotation_matrix_xi = np.array( [
        [ np.cos(q_zeta), -np.sin(q_zeta), np.zeros_like(q_zeta) ],
        [ np.sin(q_zeta),  np.cos(q_zeta), np.zeros_like(q_zeta) ],
        [ np.zeros_like(q_zeta),np.zeros_like(q_zeta),np.ones_like(q_zeta) ]
        ] )
    rotation_matrix_xi_inverse = np.swapaxes(rotation_matrix_xi,0,1)

    if Psi_3D_lab.ndim == 3: # Matrices of Psi, residing in the last index
        # To change the rotation matrices from [3,3,numberOfDataPoints] to [numberOfDataPoints,3,3]
        # Ensures that matmul will broadcast correctly
        rotation_matrix_xi = np.moveaxis(rotation_matrix_xi,-1,0)
        rotation_matrix_xi_inverse = np.moveaxis(rotation_matrix_xi_inverse,-1,0)

    Psi_3D_lab_Cartesian = np.matmul(np.matmul(rotation_matrix_xi,temp_matrix_for_Psi),rotation_matrix_xi_inverse)
    return Psi_3D_lab_Cartesian

#----------------------------------
    
# Functions (beam tracing 1)
def find_normalised_plasma_freq(electron_density, launch_angular_frequency):
    
#    if electron_density < 0:
#        print(electron_density)
#        electron_density=0
    # Electron density in units of 10^19 m-3
    normalised_plasma_freq = (constants.e*np.sqrt(electron_density*10**19 / (constants.epsilon_0*constants.m_e) )) / launch_angular_frequency
    #normalised_plasma_freq = np.sqrt(electron_density*10**19 * 3187.042702) / launch_angular_frequency # Torbeam's implementation

    return normalised_plasma_freq


def find_normalised_gyro_freq(B_Total, launch_angular_frequency):
    
    normalised_gyro_freq = constants.e * B_Total / (constants.m_e * launch_angular_frequency)   

    return normalised_gyro_freq


def find_epsilon_para(electron_density, launch_angular_frequency):
    # also called epsilon_bb in my paper
    
    normalised_plasma_freq = find_normalised_plasma_freq(electron_density, launch_angular_frequency)
    epsilon_para = 1 - normalised_plasma_freq**2
    
    return epsilon_para


def find_epsilon_perp(electron_density, B_Total, launch_angular_frequency):
    # also called epsilon_11 in my paper
   
    normalised_plasma_freq = find_normalised_plasma_freq(electron_density, launch_angular_frequency)
    normalised_gyro_freq = find_normalised_gyro_freq(B_Total, launch_angular_frequency)     
    epsilon_perp = 1 - normalised_plasma_freq**2 / (1 - normalised_gyro_freq**2)
    
    return epsilon_perp


def find_epsilon_g(electron_density, B_Total, launch_angular_frequency):
    # also called epsilon_12 in my paper
    
    normalised_plasma_freq = find_normalised_plasma_freq(electron_density, launch_angular_frequency)
    normalised_gyro_freq = find_normalised_gyro_freq(B_Total, launch_angular_frequency)     
    epsilon_g = (normalised_plasma_freq**2) * normalised_gyro_freq / (1 - normalised_gyro_freq**2)
    
    return epsilon_g
    

def find_Booker_alpha(electron_density, B_Total, sin_theta_m_sq, launch_angular_frequency):
    
    epsilon_para = find_epsilon_para(electron_density, launch_angular_frequency)
    epsilon_perp = find_epsilon_perp(electron_density, B_Total, launch_angular_frequency)
    Booker_alpha = epsilon_para*sin_theta_m_sq + epsilon_perp*(1-sin_theta_m_sq)
    
    return Booker_alpha


def find_Booker_beta(electron_density, B_Total, sin_theta_m_sq, launch_angular_frequency):
    
    epsilon_perp = find_epsilon_perp(electron_density, B_Total, launch_angular_frequency)
    epsilon_para = find_epsilon_para(electron_density, launch_angular_frequency)
    epsilon_g = find_epsilon_g(electron_density, B_Total, launch_angular_frequency)
    Booker_beta = (
            - epsilon_perp * epsilon_para * (1+sin_theta_m_sq)
            - (epsilon_perp**2 - epsilon_g**2) * (1-sin_theta_m_sq)
                   )
    
    return Booker_beta

def find_Booker_gamma(electron_density, B_Total, launch_angular_frequency):
    
    epsilon_perp = find_epsilon_perp(electron_density, B_Total, launch_angular_frequency)
    epsilon_para = find_epsilon_para(electron_density, launch_angular_frequency)
    epsilon_g = find_epsilon_g(electron_density, B_Total, launch_angular_frequency)
    Booker_gamma = epsilon_para*(epsilon_perp**2 - epsilon_g**2)
    
    return Booker_gamma
#----------------------------------
    


# Functions (beam tracing 2)
def find_H(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
           interp_poloidal_flux, find_density_1D, find_B_R, find_B_T, find_B_Z):
    # For this functions to work,  the interpolation functions for
    # electron density,  B_Total,  and psi  (poloidal flux) must be
    # declared at some point before they are called
    
    K_magnitude = np.sqrt(K_R**2 + (K_zeta/q_R)**2 + K_Z**2)
    wavenumber_K0 = launch_angular_frequency / constants.c

    poloidal_flux = interp_poloidal_flux(q_R, q_Z,grid=False)
    electron_density = find_density_1D(poloidal_flux)
    B_R = np.squeeze(find_B_R(q_R, q_Z))
    B_T = np.squeeze(find_B_T(q_R, q_Z))
    B_Z = np.squeeze(find_B_Z(q_R, q_Z))

    B_Total = np.sqrt(B_R**2 + B_T**2 + B_Z**2)
    b_hat = np.array([B_R, B_T, B_Z]) / B_Total
    K_hat = np.array([K_R, K_zeta/q_R, K_Z]) / K_magnitude
    if np.size(q_R) == 1:
        sin_theta_m_sq = (np.dot(b_hat, K_hat))**2 #square of the mismatch angle
    else: # Vectorised version of find_H
        b_hat = b_hat.T
        K_hat = K_hat.T
        sin_theta_m_sq = (contract_special(b_hat, K_hat))**2 #square of the mismatch angle
        
    Booker_alpha = find_Booker_alpha(electron_density, B_Total, sin_theta_m_sq, launch_angular_frequency)
    Booker_beta = find_Booker_beta(electron_density, B_Total, sin_theta_m_sq, launch_angular_frequency)
    Booker_gamma = find_Booker_gamma(electron_density, B_Total, launch_angular_frequency)
    
    H = (K_magnitude/wavenumber_K0)**2 + (
            Booker_beta - mode_flag *
            # np.sqrt(max(0, (Booker_beta**2 - 4*Booker_alpha*Booker_gamma)))
            np.sqrt(Booker_beta**2 - 4*Booker_alpha*Booker_gamma)
            ) / (2 * Booker_alpha)
    
    return H
    

# Functions (interface)
    # For going from vacuum to plasma (Will one day implement going from plasma to vacuum)
def find_d_poloidal_flux_dR(q_R, q_Z, delta_R, interp_poloidal_flux):
    
    poloidal_flux_0 = interp_poloidal_flux(q_R, q_Z)
    poloidal_flux_1 = interp_poloidal_flux(q_R+delta_R, q_Z) 
    poloidal_flux_2 = interp_poloidal_flux(q_R+2*delta_R, q_Z)
    d_poloidal_flux_dR = ( (-3/2)*poloidal_flux_0 + (2)*poloidal_flux_1 + (-1/2)*poloidal_flux_2 ) / (delta_R)
    
    return d_poloidal_flux_dR

def find_d_poloidal_flux_dZ(q_R, q_Z, delta_Z, interp_poloidal_flux):
    
    poloidal_flux_0 = interp_poloidal_flux(q_R, q_Z)
    poloidal_flux_1 = interp_poloidal_flux(q_R, q_Z+delta_Z)
    poloidal_flux_2 = interp_poloidal_flux(q_R, q_Z+2*delta_Z)
    d_poloidal_flux_dZ = ( (-3/2)*poloidal_flux_0 + (2)*poloidal_flux_1 + (-1/2)*poloidal_flux_2 ) / (delta_Z)
    
    return d_poloidal_flux_dZ
    
def find_Psi_3D_plasma(Psi_vacuum_3D,
                       dH_dKR, dH_dKzeta, dH_dKZ,
                       dH_dR, dH_dZ,
                       d_poloidal_flux_d_R, d_poloidal_flux_d_Z):
    # When beam is entering plasma from vacuum    
    Psi_v_R_R       = Psi_vacuum_3D[0,0]
    Psi_v_zeta_zeta = Psi_vacuum_3D[1,1]
    Psi_v_Z_Z       = Psi_vacuum_3D[2,2]
    Psi_v_R_zeta    = Psi_vacuum_3D[0,1]
    Psi_v_R_Z       = Psi_vacuum_3D[0,2]
    Psi_v_zeta_Z    = Psi_vacuum_3D[1,2]
    
    interface_matrix = np.zeros([6,6])
    interface_matrix[0][5] = 1
    interface_matrix[1][0] = d_poloidal_flux_d_Z**2
    interface_matrix[1][1] = - 2 * d_poloidal_flux_d_R * d_poloidal_flux_d_Z
    interface_matrix[1][3] = d_poloidal_flux_d_R**2
    interface_matrix[2][2] = - d_poloidal_flux_d_Z
    interface_matrix[2][4] = d_poloidal_flux_d_R
    interface_matrix[3][0] = dH_dKR
    interface_matrix[3][1] = dH_dKZ
    interface_matrix[3][2] = dH_dKzeta
    interface_matrix[4][1] = dH_dKR
    interface_matrix[4][3] = dH_dKZ
    interface_matrix[4][4] = dH_dKzeta
    interface_matrix[5][2] = dH_dKR
    interface_matrix[5][4] = dH_dKZ
    interface_matrix[5][5] = dH_dKzeta
    
    interface_matrix_inverse = np.linalg.inv(interface_matrix)
    
    [
     Psi_p_R_R, Psi_p_R_Z, Psi_p_R_zeta, 
     Psi_p_Z_Z, Psi_p_Z_zeta, Psi_p_zeta_zeta
    ] = np.matmul (interface_matrix_inverse, [
            Psi_v_zeta_zeta, 
            Psi_v_R_R * d_poloidal_flux_d_Z**2 - 2 * Psi_v_R_Z * d_poloidal_flux_d_R * d_poloidal_flux_d_Z + Psi_v_Z_Z * d_poloidal_flux_d_R **2, 
            - Psi_v_R_zeta * d_poloidal_flux_d_Z + Psi_v_zeta_Z * d_poloidal_flux_d_R, 
            -dH_dR, 
            -dH_dZ, 
            0        
           ] )

    Psi_3D_plasma = np.zeros([3,3],dtype='complex128')
    Psi_3D_plasma[0,0] = Psi_p_R_R
    Psi_3D_plasma[1,1] = Psi_p_zeta_zeta
    Psi_3D_plasma[2,2] = Psi_p_Z_Z
    Psi_3D_plasma[0,1] = Psi_p_R_zeta
    Psi_3D_plasma[1,0] = Psi_3D_plasma[0,1]
    Psi_3D_plasma[0,2] = Psi_p_R_Z
    Psi_3D_plasma[2,0] = Psi_3D_plasma[0,2]
    Psi_3D_plasma[1,2] = Psi_p_Z_zeta
    Psi_3D_plasma[2,1] = Psi_3D_plasma[1,2]
    
    return Psi_3D_plasma
# -----------------




# Functions (analysis)
    # These are not strictly necessary for beam tracing, but useful for analysis of DBS
def find_dbhat_dR(q_R, q_Z, delta_R, find_B_R, find_B_T, find_B_Z): 
    # \fract{d b_hat}{d R}
    B_R_plus_R = np.squeeze(find_B_R(q_R+delta_R, q_Z))
    B_T_plus_R = np.squeeze(find_B_T(q_R+delta_R, q_Z))
    B_Z_plus_R = np.squeeze(find_B_Z(q_R+delta_R, q_Z))

    B_R_minus_R = np.squeeze(find_B_R(q_R-delta_R, q_Z))
    B_T_minus_R = np.squeeze(find_B_T(q_R-delta_R, q_Z))
    B_Z_minus_R = np.squeeze(find_B_Z(q_R-delta_R, q_Z))
    
    B_magnitude_plus = np.sqrt(B_R_plus_R**2 + B_T_plus_R**2 + B_Z_plus_R**2)
    b_hat_plus = np.array([B_R_plus_R, B_T_plus_R, B_Z_plus_R]) / B_magnitude_plus
    
    B_magnitude_minus = np.sqrt(B_R_minus_R**2 + B_T_minus_R**2 + B_Z_minus_R**2)
    b_hat_minus = np.array([B_R_minus_R, B_T_minus_R, B_Z_minus_R]) / B_magnitude_minus    
    
    dbhat_dR = (b_hat_plus - b_hat_minus) / (2 * delta_R)
        
    return dbhat_dR

def find_dbhat_dZ(q_R, q_Z, delta_Z, find_B_R, find_B_T, find_B_Z): 
    # \fract{d b_hat}{d R}
    B_R_plus_Z = np.squeeze(find_B_R(q_R, q_Z+delta_Z))
    B_T_plus_Z = np.squeeze(find_B_T(q_R, q_Z+delta_Z))
    B_Z_plus_Z = np.squeeze(find_B_Z(q_R, q_Z+delta_Z))

    B_R_minus_Z = np.squeeze(find_B_R(q_R, q_Z-delta_Z,))
    B_T_minus_Z = np.squeeze(find_B_T(q_R, q_Z-delta_Z,))
    B_Z_minus_Z = np.squeeze(find_B_Z(q_R, q_Z-delta_Z,))
    
    B_magnitude_plus = np.sqrt(B_R_plus_Z**2 + B_T_plus_Z**2 + B_Z_plus_Z**2)
    b_hat_plus = np.array([B_R_plus_Z, B_T_plus_Z, B_Z_plus_Z]) / B_magnitude_plus
    
    B_magnitude_minus = np.sqrt(B_R_minus_Z**2 + B_T_minus_Z**2 + B_Z_minus_Z**2)
    b_hat_minus = np.array([B_R_minus_Z, B_T_minus_Z, B_Z_minus_Z]) / B_magnitude_minus    
    
    dbhat_dZ = (b_hat_plus - b_hat_minus) / (2 * delta_Z)
        
    return dbhat_dZ

def find_Rayleigh_length(waist, wavenumber): #Finds the size of the waist (assumes vacuum propagation and circular beam)
    Rayleigh_length = 0.5 * wavenumber * waist**2
    return Rayleigh_length

def find_waist(width, wavenumber, curvature): #Finds the size of the waist (assumes vacuum propagation and circular beam)
    waist = width / np.sqrt(1+curvature**2 * width**4 * wavenumber**2 / 4)
    return waist

def find_distance_from_waist(width, wavenumber, curvature): #Finds how far you are from the waist (assumes vacuum propagation and circular beam)
    waist = width / np.sqrt(1+curvature**2 * width**4 * wavenumber**2 / 4)
    distance_from_waist = np.sign(curvature)*np.sqrt((width**2 - waist**2)*waist**2*wavenumber**2/4)
    return distance_from_waist

#def find_g_magnitude(q_R,q_Z,K_R,K_zeta,K_Z,launch_angular_frequency,mode_flag,delta_K_R,delta_K_zeta,delta_K_Z,
#                     interp_poloidal_flux,find_density_1D,find_B_R,find_B_T,find_B_Z): # Finds the magnitude of the group velocity. This method is slow, do not use in main loop.\
#    dH_dKR   = find_dH_dKR(
#                           q_R,q_Z,
#                           K_R,K_zeta,K_Z,
#                           launch_angular_frequency,mode_flag,delta_K_R,
#                           interp_poloidal_flux,find_density_1D,
#                           find_B_R,find_B_T,find_B_Z
#                          )
#    dH_dKzeta = find_dH_dKzeta(
#                               q_R,q_Z,
#                               K_R,K_zeta,K_Z,
#                               launch_angular_frequency,mode_flag,delta_K_zeta,
#                               interp_poloidal_flux,find_density_1D,
#                               find_B_R,find_B_T,find_B_Z
#                              )
#    dH_dKZ    = find_dH_dKZ(
#                            q_R,q_Z,
#                            K_R,K_zeta,K_Z,
#                            launch_angular_frequency,mode_flag,delta_K_Z,
#                            interp_poloidal_flux,find_density_1D,
#                            find_B_R,find_B_T,find_B_Z
#                           )    
#    g_magnitude = (q_R**2 * dH_dKzeta**2 + dH_dKR**2 + dH_dKZ**2)**0.5       
#    return g_magnitude

def find_D(K_magnitude,launch_angular_frequency,epsilon_para,epsilon_perp,epsilon_g,theta_m): 
    # Finds the dispersion
    
    wavenumber_K0 = launch_angular_frequency / constants.c
    n_ref_index = K_magnitude/wavenumber_K0
    sin_theta_m = np.sin(theta_m)
    cos_theta_m = np.cos(theta_m)
    
    
    D_11_component = epsilon_perp - n_ref_index**2*sin_theta_m**2
    D_22_component = epsilon_perp - n_ref_index**2
    D_bb_component = epsilon_para - n_ref_index**2*cos_theta_m**2
    D_12_component = epsilon_g
    D_1b_component = n_ref_index**2*sin_theta_m*cos_theta_m
    
    return D_11_component, D_22_component, D_bb_component, D_12_component, D_1b_component
    

def find_H_Cardano(K_magnitude,launch_angular_frequency,epsilon_para,epsilon_perp,epsilon_g,theta_m):  
    # This function is designed to be evaluated in post-procesing, hence it uses different inputs from the usual H
    
    D_11_component, D_22_component, D_bb_component, D_12_component, D_1b_component = find_D(K_magnitude,launch_angular_frequency,epsilon_para,epsilon_perp,epsilon_g,theta_m)
    
    h_2_coefficient = - D_11_component - D_22_component - D_bb_component
    h_1_coefficient = D_11_component*D_bb_component + D_11_component*D_22_component + D_22_component*D_bb_component - D_12_component**2 - D_1b_component**2
    h_0_coefficient = D_22_component*D_1b_component**2 + D_bb_component*D_12_component**2 - D_11_component*D_22_component*D_bb_component
    
    h_t_coefficient = (
                            -2*h_2_coefficient**3
                            +9*h_2_coefficient*h_1_coefficient
                            -27*h_0_coefficient
                            +3*np.sqrt(3)*np.sqrt(
                                4*h_2_coefficient**3 * h_0_coefficient
                                -h_2_coefficient**2 * h_1_coefficient**2
                                -18*h_2_coefficient * h_1_coefficient * h_0_coefficient
                                +4*h_1_coefficient**3
                                +27*h_0_coefficient**2
                                +0j #to make the argument of the np.sqrt complex, so that the sqrt evaluates negative functions
                            )
                        )**(1/3)
    
    H_1_Cardano = h_t_coefficient/(3*2**(1/3)) - 2**(1/3) *(3*h_1_coefficient - h_2_coefficient**2)/(3*h_t_coefficient) - h_2_coefficient/3
    H_2_Cardano = - (1 - 1j*np.sqrt(3))/(6*2**(1/3))*h_t_coefficient + (1 + 1j*np.sqrt(3))*(3*h_1_coefficient - h_2_coefficient**2)/(3*2**(2/3)*h_t_coefficient) - h_2_coefficient/3
    H_3_Cardano = - (1 + 1j*np.sqrt(3))/(6*2**(1/3))*h_t_coefficient + (1 - 1j*np.sqrt(3))*(3*h_1_coefficient - h_2_coefficient**2)/(3*2**(2/3)*h_t_coefficient) - h_2_coefficient/3
    return H_1_Cardano, H_2_Cardano, H_3_Cardano

def find_widths_and_curvatures(Psi_xx, Psi_xy, Psi_yy, K_magnitude):
    Psi_w_real = np.array(np.real([
                    [Psi_xx,Psi_xy],
                    [Psi_xy,Psi_yy]
                    ]))
    Psi_w_imag = np.array(np.imag([
                        [Psi_xx,Psi_xy],
                        [Psi_xy,Psi_yy]
                        ]))
    
    Psi_w_real_eigvals, Psi_w_real_eigvecs = np.linalg.eig(Psi_w_real)
    Psi_w_imag_eigvals, Psi_w_imag_eigvecs = np.linalg.eig(Psi_w_imag)
    #The normalized (unit “length”) eigenvectors, such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    
    widths = np.sqrt(2/Psi_w_imag_eigvals)
    curvatures = Psi_w_real_eigvals/K_magnitude # curvature = 1/radius_of_curvature
    
    return widths, Psi_w_imag_eigvecs, curvatures, Psi_w_real_eigvecs
#----------------------------------
    


# Functions (Debugging)

def find_dB_dR_CFD(q_R, q_Z, delta_R, find_B_R, find_B_T, find_B_Z): 
    """
    Finds \fract{d B}{d Z}, where B is the magnitude of the B field
    """    
    B_R_plus_R = np.squeeze(find_B_R(q_R+delta_R, q_Z))
    B_T_plus_R = np.squeeze(find_B_T(q_R+delta_R, q_Z))
    B_Z_plus_R = np.squeeze(find_B_Z(q_R+delta_R, q_Z))

    B_R_minus_R = np.squeeze(find_B_R(q_R-delta_R, q_Z))
    B_T_minus_R = np.squeeze(find_B_T(q_R-delta_R, q_Z))
    B_Z_minus_R = np.squeeze(find_B_Z(q_R-delta_R, q_Z))
    
    B_magnitude_plus = np.sqrt(B_R_plus_R**2 + B_T_plus_R**2 + B_Z_plus_R**2)
    
    B_magnitude_minus = np.sqrt(B_R_minus_R**2 + B_T_minus_R**2 + B_Z_minus_R**2)
    
    dB_dZ = (B_magnitude_plus - B_magnitude_minus) / (2 * delta_R)    
    return dB_dZ

def find_dB_dZ_CFD(q_R, q_Z, delta_Z, find_B_R, find_B_T, find_B_Z): 
    """
    Finds \fract{d B}{d Z}, where B is the magnitude of the B field
    """    
    B_R_plus_Z = np.squeeze(find_B_R(q_R, q_Z+delta_Z))
    B_T_plus_Z = np.squeeze(find_B_T(q_R, q_Z+delta_Z))
    B_Z_plus_Z = np.squeeze(find_B_Z(q_R, q_Z+delta_Z))

    B_R_minus_Z = np.squeeze(find_B_R(q_R, q_Z-delta_Z,))
    B_T_minus_Z = np.squeeze(find_B_T(q_R, q_Z-delta_Z,))
    B_Z_minus_Z = np.squeeze(find_B_Z(q_R, q_Z-delta_Z,))
    
    B_magnitude_plus = np.sqrt(B_R_plus_Z**2 + B_T_plus_Z**2 + B_Z_plus_Z**2)
    
    B_magnitude_minus = np.sqrt(B_R_minus_Z**2 + B_T_minus_Z**2 + B_Z_minus_Z**2)
    
    dB_dZ = (B_magnitude_plus - B_magnitude_minus) / (2 * delta_Z)    
    return dB_dZ

def find_d2B_dR2_CFD(q_R, q_Z, delta_R, find_B_R, find_B_T, find_B_Z): 
    """
    Finds \fract{d^2 B}{d R^2}, where B is the magnitude of the B field
    """    
    B_R_0 = np.squeeze(find_B_R(q_R, q_Z))
    B_T_0 = np.squeeze(find_B_T(q_R, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_R, q_Z))
    B_magnitude_0 = np.sqrt(B_R_0**2 + B_T_0**2 + B_Z_0**2)
    
    B_R_plus = np.squeeze(find_B_R(q_R+delta_R, q_Z))
    B_T_plus = np.squeeze(find_B_T(q_R+delta_R, q_Z))
    B_Z_plus = np.squeeze(find_B_Z(q_R+delta_R, q_Z))
    B_magnitude_plus = np.sqrt(B_R_plus**2 + B_T_plus**2 + B_Z_plus**2)
    
    B_R_minus = np.squeeze(find_B_R(q_R-delta_R, q_Z))
    B_T_minus = np.squeeze(find_B_T(q_R-delta_R, q_Z))
    B_Z_minus = np.squeeze(find_B_Z(q_R-delta_R, q_Z))
    B_magnitude_minus = np.sqrt(B_R_minus**2 + B_T_minus**2 + B_Z_minus**2)    
    
    d2B_dR2 = ( (1)*B_magnitude_minus + (-2)*B_magnitude_0 + (1)*B_magnitude_plus ) / (delta_R**2)
    return d2B_dR2


def find_d2B_dZ2_CFD(q_R, q_Z, delta_Z, find_B_R, find_B_T, find_B_Z):
    
    B_R_0 = np.squeeze(find_B_R(q_R, q_Z))
    B_T_0 = np.squeeze(find_B_T(q_R, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_R, q_Z))
    B_magnitude_0 = np.sqrt(B_R_0**2 + B_T_0**2 + B_Z_0**2)
    
    B_R_plus = np.squeeze(find_B_R(q_R, q_Z+delta_Z))
    B_T_plus = np.squeeze(find_B_T(q_R, q_Z+delta_Z))
    B_Z_plus = np.squeeze(find_B_Z(q_R, q_Z+delta_Z))
    B_magnitude_plus = np.sqrt(B_R_plus**2 + B_T_plus**2 + B_Z_plus**2)
    
    B_R_minus = np.squeeze(find_B_R(q_R, q_Z-delta_Z))
    B_T_minus = np.squeeze(find_B_T(q_R, q_Z-delta_Z))
    B_Z_minus = np.squeeze(find_B_Z(q_R, q_Z-delta_Z))
    B_magnitude_minus = np.sqrt(B_R_minus**2 + B_T_minus**2 + B_Z_minus**2)    
    
    d2B_dZ2 = ( (1)*B_magnitude_minus + (-2)*B_magnitude_0 + (1)*B_magnitude_plus ) / (delta_Z**2)    
    return d2B_dZ2

#def find_d2B_dR_dZ_CFD(q_R, q_Z, delta_R, delta_Z, find_B_R, find_B_T, find_B_Z):
#    
#    dB_dZ_0 = find_dB_dZ_FFD(q_R, 
#                             q_Z, delta_Z, find_B_R, find_B_T, find_B_Z)
#    dB_dZ_1 = find_dB_dZ_FFD(q_R+delta_R, 
#                             q_Z, delta_Z, find_B_R, find_B_T, find_B_Z)
#    dB_dZ_2 = find_dB_dZ_FFD(q_R+2*delta_R, 
#                             q_Z, delta_Z, find_B_R, find_B_T, find_B_Z)
#    d2B_dR_dZ = ( (-3/2)*dB_dZ_0 + (2)*dB_dZ_1 + (-1/2)*dB_dZ_2 ) / (delta_R)
#    
##    return d2B_dR_dZ
#    
#    
#    
#    
#    
#    
#    B_plus_R_plus_Z   = find_H(q_R+delta_R, q_Z+delta_Z)
#    B_plus_R_minus_Z  = find_H(q_R+delta_R, q_Z-delta_Z) 
#    B_minus_R_plus_Z  = find_H(q_R-delta_R, q_Z+delta_Z)
#    B_minus_R_minus_Z = find_H(q_R-delta_R, q_Z-delta_Z)
#    d2B_dR_dZ = (B_plus_R_plus_Z - B_plus_R_minus_Z - B_minus_R_plus_Z + B_minus_R_minus_Z) / (4 * delta_R * delta_Z)    
    
    

def find_dB_dR_FFD(q_R, q_Z, delta_R, find_B_R, find_B_T, find_B_Z): 
    """
    Finds \fract{d B}{d Z}, where B is the magnitude of the B field
    """    
    B_R_0 = np.squeeze(find_B_R(q_R, q_Z))
    B_T_0 = np.squeeze(find_B_T(q_R, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_R, q_Z))
    B_magnitude_0 = np.sqrt(B_R_0**2 + B_T_0**2 + B_Z_0**2)
    
    B_R_1 = np.squeeze(find_B_R(q_R+delta_R, q_Z))
    B_T_1 = np.squeeze(find_B_T(q_R+delta_R, q_Z))
    B_Z_1 = np.squeeze(find_B_Z(q_R+delta_R, q_Z))
    B_magnitude_1 = np.sqrt(B_R_1**2 + B_T_1**2 + B_Z_1**2)
    
    B_R_2 = np.squeeze(find_B_R(q_R+2*delta_R, q_Z))
    B_T_2 = np.squeeze(find_B_T(q_R+2*delta_R, q_Z))
    B_Z_2 = np.squeeze(find_B_Z(q_R+2*delta_R, q_Z))
    B_magnitude_2 = np.sqrt(B_R_2**2 + B_T_2**2 + B_Z_2**2)    
    
    dB_dR = ( (-3/2)*B_magnitude_0 + (2)*B_magnitude_1 + (-1/2)*B_magnitude_2 ) / (delta_R)
    return dB_dR

def find_dB_dZ_FFD(q_R, q_Z, delta_Z, find_B_R, find_B_T, find_B_Z): 
    """
    Finds \fract{d B}{d Z}, where B is the magnitude of the B field
    """    
    B_R_0 = np.squeeze(find_B_R(q_R, q_Z))
    B_T_0 = np.squeeze(find_B_T(q_R, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_R, q_Z))
    B_magnitude_0 = np.sqrt(B_R_0**2 + B_T_0**2 + B_Z_0**2)
    
    B_R_1 = np.squeeze(find_B_R(q_R, q_Z+delta_Z))
    B_T_1 = np.squeeze(find_B_T(q_R, q_Z+delta_Z))
    B_Z_1 = np.squeeze(find_B_Z(q_R, q_Z+delta_Z))
    B_magnitude_1 = np.sqrt(B_R_1**2 + B_T_1**2 + B_Z_1**2)
    
    B_R_2 = np.squeeze(find_B_R(q_R, q_Z+2*delta_Z))
    B_T_2 = np.squeeze(find_B_T(q_R, q_Z+2*delta_Z))
    B_Z_2 = np.squeeze(find_B_Z(q_R, q_Z+2*delta_Z))
    B_magnitude_2 = np.sqrt(B_R_2**2 + B_T_2**2 + B_Z_2**2)    
    
    dB_dZ = ( (-3/2)*B_magnitude_0 + (2)*B_magnitude_1 + (-1/2)*B_magnitude_2 ) / (delta_Z)
    return dB_dZ

def find_d2B_dR2_FFD(q_R, q_Z, delta_R, find_B_R, find_B_T, find_B_Z): 
    """
    Finds \fract{d B}{d Z}, where B is the magnitude of the B field
    """    
    B_R_0 = np.squeeze(find_B_R(q_R, q_Z))
    B_T_0 = np.squeeze(find_B_T(q_R, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_R, q_Z))
    B_magnitude_0 = np.sqrt(B_R_0**2 + B_T_0**2 + B_Z_0**2)
    
    B_R_1 = np.squeeze(find_B_R(q_R+delta_R, q_Z))
    B_T_1 = np.squeeze(find_B_T(q_R+delta_R, q_Z))
    B_Z_1 = np.squeeze(find_B_Z(q_R+delta_R, q_Z))
    B_magnitude_1 = np.sqrt(B_R_1**2 + B_T_1**2 + B_Z_1**2)
    
    B_R_2 = np.squeeze(find_B_R(q_R+2*delta_R, q_Z))
    B_T_2 = np.squeeze(find_B_T(q_R+2*delta_R, q_Z))
    B_Z_2 = np.squeeze(find_B_Z(q_R+2*delta_R, q_Z))
    B_magnitude_2 = np.sqrt(B_R_2**2 + B_T_2**2 + B_Z_2**2)    

    B_R_3 = np.squeeze(find_B_R(q_R+3*delta_R, q_Z))
    B_T_3 = np.squeeze(find_B_T(q_R+3*delta_R, q_Z))
    B_Z_3 = np.squeeze(find_B_Z(q_R+3*delta_R, q_Z))
    B_magnitude_3 = np.sqrt(B_R_3**2 + B_T_3**2 + B_Z_3**2)  
    
    d2B_dR2 = ( (2)*B_magnitude_0 + (-5)*B_magnitude_1 + (4)*B_magnitude_2 + (-1)*B_magnitude_3 ) / (delta_R**2)
    return d2B_dR2

def find_d2B_dZ2_FFD(q_R, q_Z, delta_Z, find_B_R, find_B_T, find_B_Z):
    
    B_R_0 = np.squeeze(find_B_R(q_R, q_Z))
    B_T_0 = np.squeeze(find_B_T(q_R, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_R, q_Z))
    B_magnitude_0 = np.sqrt(B_R_0**2 + B_T_0**2 + B_Z_0**2)
    
    B_R_1 = np.squeeze(find_B_R(q_R, q_Z+delta_Z))
    B_T_1 = np.squeeze(find_B_T(q_R, q_Z+delta_Z))
    B_Z_1 = np.squeeze(find_B_Z(q_R, q_Z+delta_Z))
    B_magnitude_1 = np.sqrt(B_R_1**2 + B_T_1**2 + B_Z_1**2)
    
    B_R_2 = np.squeeze(find_B_R(q_R, q_Z+2*delta_Z))
    B_T_2 = np.squeeze(find_B_T(q_R, q_Z+2*delta_Z))
    B_Z_2 = np.squeeze(find_B_Z(q_R, q_Z+2*delta_Z))
    B_magnitude_2 = np.sqrt(B_R_2**2 + B_T_2**2 + B_Z_2**2)  
    
    B_R_3 = np.squeeze(find_B_R(q_R, q_Z+3*delta_Z))
    B_T_3 = np.squeeze(find_B_T(q_R, q_Z+3*delta_Z))
    B_Z_3 = np.squeeze(find_B_Z(q_R, q_Z+3*delta_Z))
    B_magnitude_3 = np.sqrt(B_R_3**2 + B_T_3**2 + B_Z_3**2)  
    
    d2B_dZ2 = ( (2)*B_magnitude_0 + (-5)*B_magnitude_1 + (4)*B_magnitude_2 + (-1)*B_magnitude_3 ) / (delta_Z**2)
    
    return d2B_dZ2

def find_d2B_dR_dZ_FFD(q_R, q_Z, delta_R, delta_Z, find_B_R, find_B_T, find_B_Z):
    
    dB_dZ_0 = find_dB_dZ_FFD(q_R, 
                             q_Z, delta_Z, find_B_R, find_B_T, find_B_Z)
    dB_dZ_1 = find_dB_dZ_FFD(q_R+delta_R, 
                             q_Z, delta_Z, find_B_R, find_B_T, find_B_Z)
    dB_dZ_2 = find_dB_dZ_FFD(q_R+2*delta_R, 
                             q_Z, delta_Z, find_B_R, find_B_T, find_B_Z)
    d2B_dR_dZ = ( (-3/2)*dB_dZ_0 + (2)*dB_dZ_1 + (-1/2)*dB_dZ_2 ) / (delta_R)
    
    return d2B_dR_dZ


def find_d2_poloidal_flux_dR2(q_R, q_Z, delta_R, interp_poloidal_flux):
    
    poloidal_flux_0 = interp_poloidal_flux(q_R          , q_Z)
    poloidal_flux_1 = interp_poloidal_flux(q_R+  delta_R, q_Z) 
    poloidal_flux_2 = interp_poloidal_flux(q_R+2*delta_R, q_Z)
    poloidal_flux_3 = interp_poloidal_flux(q_R+3*delta_R, q_Z)
    d2_poloidal_flux_dR2 = ( (2)*poloidal_flux_0 + (-5)*poloidal_flux_1 + (4)*poloidal_flux_2 + (-1)*poloidal_flux_3 ) / (delta_R**2)
    
    return d2_poloidal_flux_dR2


def find_d2_poloidal_flux_dZ2(q_R, q_Z, delta_Z, interp_poloidal_flux):
    
    poloidal_flux_0 = interp_poloidal_flux(q_R, q_Z          )
    poloidal_flux_1 = interp_poloidal_flux(q_R, q_Z+  delta_Z) 
    poloidal_flux_2 = interp_poloidal_flux(q_R, q_Z+2*delta_Z)
    poloidal_flux_3 = interp_poloidal_flux(q_R, q_Z+3*delta_Z)
    d2_poloidal_flux_dZ2 = ( (2)*poloidal_flux_0 + (-5)*poloidal_flux_1 + (4)*poloidal_flux_2 + (-1)*poloidal_flux_3 ) / (delta_Z**2)
    
    return d2_poloidal_flux_dZ2

#----------------------------------



# Functions (Launch angle)
"""
Written by Neal Crocker, added to Scotty by Valerian.
Converts mirror angles of the MAST DBS to launch angles (genray)
Genray -> Scotty/Torbeam: poloidal launch angle has opposite sign
"""
def use_deg_args(func,*args,**kwargs):
    'transform args from degrees to radians before calling' #doc string
    return func(*[a*np.pi/180 for a in args],**kwargs)

def tilt_trns_RZ_make(t):
    'tilts R ([1,0,0]) in to Z ([0,0,1]) using angle in radians' #doc string
    ctilt=np.cos(t)
    stilt=np.sin(t)
    return np.array([[ctilt,0,stilt],[0,1,0],[-stilt,0,ctilt]])

def tilt_trns_TZ_make(t):
    'tilts T ([1,0,0]) in to Z ([0,0,1]) using angle in radians' #doc string
    ctilt=np.cos(t)
    stilt=np.sin(t)
    return np.array([[1,0,0],[0,ctilt,-stilt],[0,stilt,ctilt]])

def rot_trns_make(r): 
    'rotates R ([1,0,0]) into T ([0,1,0]) using angle in radians' #doc string
    crot=np.cos(r)
    srot=np.sin(r)
    return np.array([[crot,-srot,0],[srot,crot,0],[0,0,1]])

def mirrornorm_make_with_rot_tilt_operators(r,t):
    'makes norm vector for mirror using rot and tilt operators given angles in radians' #doc string
    mirrornorm0=np.array([0,1,0])
    tilt_trns=tilt_trns_TZ_make(t)
    rot_trns=rot_trns_make(np.pi/4+r)
    mirrornorm = rot_trns @ (tilt_trns @ mirrornorm0)
    return mirrornorm

def mirrornorm_make(r,t):
    'makes norm vector for mirror directly using angles in radians' #doc string
    mirrornorm=np.array([*(np.cos(t)*np.array([-np.sin(r+np.pi/4),np.cos(r+np.pi/4)])),np.sin(t)])
    return mirrornorm

def reflector_make_from_mirrornorm(mirrornorm):
    'makes relector matrix for mirror using norm vector for mirror' #doc string
    reflector=np.eye(3)-2*mirrornorm[:,np.newaxis]@mirrornorm[np.newaxis,:]
    return reflector
    
def reflector_make(r,t):
    'makes relector matrix for mirror using angles in radians' #doc string
    mirrornorm = mirrornorm_make(r,t)
    reflector=reflector_make_from_mirrornorm(mirrornorm)
    return reflector

def genray_angles_from_mirror_angles(rot_ang_deg,tilt_ang_deg,offset_for_window_norm_to_R = 3.0):
    '''
      Input args are steering mirror rotation and tilt angles in degrees (rot_ang_deg,tilt_ang_deg).
      rot_ang_deg,tilt_ang_deg = 0,0 imply beam propagates in negative window normal direction.
      optional keyword offset_for_window_norm_to_R gives toroial angle (in degrees) of 
      window normal to major radial direction, so for rot_ang_deg,tilt_ang_deg = 0,0, beam would have
      toroidal angle given by offset_for_window_norm_to_R.
    
      can get from rot_ang_deg,tilt_ang_deg buy using toroidal and poloidal "genray angles" (tor_ang, pol_ang)
      in IDL savefile log (/u/jhilles/DBS/dbs_logbook_2013.idl).  The angles were calculated according to the logic: 
          tor_ang = 3.0 - rot_ang_deg
          pol_ang = tilt_ang_deg
    '''
    
    
    beam0=np.array([0,-1,0])
    rdeg=rot_ang_deg if np.isscalar(rot_ang_deg) else  rot_ang_deg.flat[0]
    tdeg=tilt_ang_deg if np.isscalar(tilt_ang_deg) else  tilt_ang_deg.flat[0]
    reflector=use_deg_args(reflector_make,rdeg,tdeg)
    beam_windowframe = reflector @ beam0
    #rotate from window frame to port frame:
    beam = use_deg_args(rot_trns_make,-offset_for_window_norm_to_R) @ beam_windowframe

    tor_ang_genray=np.arctan2(beam[1],-beam[0])#+offset_for_window_norm_to_R*np.pi/180
    pol_ang_genray=np.arctan2(beam[2],np.sqrt(beam[0]**2+beam[1]**2))
    tor_ang_genray_deg,pol_ang_genray_deg = tor_ang_genray*180/np.pi,pol_ang_genray*180/np.pi

    return tor_ang_genray_deg,pol_ang_genray_deg