# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: chenv
Valerian Hongjie Hall-Chen
valerian.chen@gmail.com

Run in Python 3,  does not work in Python 2
"""

import numpy as np
from scipy import constants as constants

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
    
    normalised_plasma_freq = find_normalised_plasma_freq(electron_density, launch_angular_frequency)
    epsilon_para = 1 - normalised_plasma_freq**2
    
    return epsilon_para


def find_epsilon_perp(electron_density, B_Total, launch_angular_frequency):
    
    normalised_plasma_freq = find_normalised_plasma_freq(electron_density, launch_angular_frequency)
    normalised_gyro_freq = find_normalised_gyro_freq(B_Total, launch_angular_frequency)     
    epsilon_perp = 1 - normalised_plasma_freq**2 / (1 - normalised_gyro_freq**2)
    
    return epsilon_perp


def find_epsilon_g(electron_density, B_Total, launch_angular_frequency):
    
    normalised_plasma_freq = find_normalised_plasma_freq(electron_density, launch_angular_frequency)
    normalised_gyro_freq = find_normalised_gyro_freq(B_Total, launch_angular_frequency)     
    epsilon_g = - (normalised_plasma_freq**2) * normalised_gyro_freq / (1 - normalised_gyro_freq**2)
    
    return epsilon_g
    

def find_Booker_alpha(electron_density, B_Total, cos_theta_sq, launch_angular_frequency):
    
    epsilon_para = find_epsilon_para(electron_density, launch_angular_frequency)
    epsilon_perp = find_epsilon_perp(electron_density, B_Total, launch_angular_frequency)
    Booker_alpha = epsilon_para*cos_theta_sq + epsilon_perp*(1-cos_theta_sq)
    
    return Booker_alpha


def find_Booker_beta(electron_density, B_Total, cos_theta_sq, launch_angular_frequency):
    
    epsilon_perp = find_epsilon_perp(electron_density, B_Total, launch_angular_frequency)
    epsilon_para = find_epsilon_para(electron_density, launch_angular_frequency)
    epsilon_g = find_epsilon_g(electron_density, B_Total, launch_angular_frequency)
    Booker_beta = (
            - epsilon_perp * epsilon_para * (1+cos_theta_sq)
            - (epsilon_perp**2 - epsilon_g**2) * (1-cos_theta_sq)
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
           interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    # For this functions to work,  the interpolation functions for
    # electron density,  B_Total,  and psi  (poloidal flux) must be
    # declared at some point before they are called
    
    K_magnitude = np.sqrt(K_R**2 + (K_zeta/q_R)**2 + K_Z**2)
    wavenumber_K0 = launch_angular_frequency / constants.c

    poloidal_flux = interp_poloidal_flux(q_R, q_Z)    
    electron_density = interp_density_1D(poloidal_flux)
    B_R = np.squeeze(interp_B_R(q_R, q_Z))
    B_T = np.squeeze(interp_B_T(q_R, q_Z))
    B_Z = np.squeeze(interp_B_Z(q_R, q_Z))
    
    B_Total = np.sqrt(B_R**2 + B_T**2 + B_Z**2)
    b_hat = np.array([B_R, B_T, B_Z]) / B_Total
    K_hat = np.array([K_R, K_zeta/q_R, K_Z]) / K_magnitude
    cos_theta_sq = (np.dot(b_hat, K_hat))**2
 
    Booker_alpha = find_Booker_alpha(electron_density, B_Total, cos_theta_sq, launch_angular_frequency)
    Booker_beta = find_Booker_beta(electron_density, B_Total, cos_theta_sq, launch_angular_frequency)
    Booker_gamma = find_Booker_gamma(electron_density, B_Total, launch_angular_frequency)
    
    H = (K_magnitude/wavenumber_K0)**2 + (
            Booker_beta - mode_flag *
            np.sqrt(max(0, (Booker_beta**2 - 4*Booker_alpha*Booker_gamma)))
            ) / (2 * Booker_alpha)
    
    return H
    

def find_dH_dR(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_R, 
               interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus  = find_H(q_R+delta_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                     interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus = find_H(q_R-delta_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                     interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    dH_dR = (H_plus - H_minus) / (2 * delta_R)
    return dH_dR


def find_dH_dZ(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_Z, 
               interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus  = find_H(q_R, q_Z+delta_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                     interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus = find_H(q_R, q_Z-delta_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                     interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    dH_dZ = (H_plus - H_minus) / (2 * delta_Z)
    
    return dH_dZ


def find_dH_dKR(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_R, 
                interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus  = find_H(q_R, q_Z, K_R+delta_K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                     interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus = find_H(q_R, q_Z, K_R-delta_K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                     interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    dH_dKR = (H_plus - H_minus) / (2 * delta_K_R)
    
    return dH_dKR


def find_dH_dKZ(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_Z, 
                interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus  = find_H(q_R, q_Z, K_R, K_zeta, K_Z+delta_K_Z, launch_angular_frequency, mode_flag, 
                     interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus = find_H(q_R, q_Z, K_R, K_zeta, K_Z-delta_K_Z, launch_angular_frequency, mode_flag, 
                     interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    dH_dKZ = (H_plus - H_minus) / (2 * delta_K_Z)
    
    return dH_dKZ


def find_dH_dKzeta(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_zeta, 
                   interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus  = find_H(q_R, q_Z, K_R, K_zeta+delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                     interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus = find_H(q_R, q_Z, K_R, K_zeta-delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                     interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    dH_dKzeta = (H_plus - H_minus) / (2 * delta_K_zeta)
    
    return dH_dKzeta
      

def find_d2H_dR2(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_R, 
                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus    = find_H(q_R+delta_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_current = find_H(q_R,         q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus   = find_H(q_R-delta_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    d2H_dR2 = (H_plus - 2*H_current + H_minus) / (delta_R**2)
    
    return d2H_dR2


def find_d2H_dZ2(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_Z, 
                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus    = find_H(q_R, q_Z+delta_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_current = find_H(q_R, q_Z,         K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus   = find_H(q_R, q_Z-delta_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    d2H_dZ2 = (H_plus - 2*H_current + H_minus) / (delta_Z**2)
    
    return d2H_dZ2


def find_d2H_dR_dZ(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_R, delta_Z, 
                   interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus_R_plus_Z   = find_H(q_R+delta_R, q_Z+delta_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                               interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_plus_R_minus_Z  = find_H(q_R+delta_R, q_Z-delta_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                               interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z) 
    H_minus_R_plus_Z  = find_H(q_R-delta_R, q_Z+delta_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                               interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_R_minus_Z = find_H(q_R-delta_R, q_Z-delta_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                               interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    d2H_dR_dZ = (H_plus_R_plus_Z - H_plus_R_minus_Z - H_minus_R_plus_Z + H_minus_R_minus_Z) / (4 * delta_R * delta_Z)
    
    return d2H_dR_dZ


def find_d2H_dKR2(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_R, 
                  interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus    = find_H(q_R, q_Z, K_R+delta_K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_current = find_H(q_R, q_Z, K_R,           K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus   = find_H(q_R, q_Z, K_R-delta_K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    d2H_dKR2 = (H_plus - 2*H_current + H_minus) / (delta_K_R**2)
    
    return d2H_dKR2


def find_d2H_dKR_dKzeta(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_R, delta_K_zeta, 
                        interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus_K_R_plus_K_zeta   = find_H(q_R, q_Z, K_R+delta_K_R, K_zeta+delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                    interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_plus_K_R_minus_K_zeta  = find_H(q_R, q_Z, K_R+delta_K_R, K_zeta-delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                    interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_R_plus_K_zeta  = find_H(q_R, q_Z, K_R-delta_K_R, K_zeta+delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                    interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_R_minus_K_zeta = find_H(q_R, q_Z, K_R-delta_K_R, K_zeta-delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                    interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    d2H_dKR_dKzeta = (H_plus_K_R_plus_K_zeta - H_plus_K_R_minus_K_zeta - H_minus_K_R_plus_K_zeta + H_minus_K_R_minus_K_zeta) / (4*delta_K_R*delta_K_zeta)
    
    return d2H_dKR_dKzeta


def find_d2H_dKR_dKz(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_R, delta_K_Z, 
                     interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus_K_R_plus_K_z   = find_H(q_R, q_Z, K_R+delta_K_R, K_zeta, K_Z+delta_K_Z, launch_angular_frequency, mode_flag, 
                                   interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_plus_K_R_minus_K_z  = find_H(q_R, q_Z, K_R+delta_K_R, K_zeta, K_Z-delta_K_Z, launch_angular_frequency, mode_flag, 
                                   interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_R_plus_K_z  = find_H(q_R, q_Z, K_R-delta_K_R, K_zeta, K_Z+delta_K_Z, launch_angular_frequency, mode_flag, 
                                   interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_R_minus_K_z = find_H(q_R, q_Z, K_R-delta_K_R, K_zeta, K_Z-delta_K_Z, launch_angular_frequency, mode_flag, 
                                   interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    d2H_dKR_dKz = (H_plus_K_R_plus_K_z - H_plus_K_R_minus_K_z - H_minus_K_R_plus_K_z + H_minus_K_R_minus_K_z) / (4*delta_K_R*delta_K_Z)
    
    return d2H_dKR_dKz


def find_d2H_dKzeta2(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_zeta, 
                     interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus    = find_H(q_R, q_Z, K_R, K_zeta+delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_current = find_H(q_R, q_Z, K_R, K_zeta,            K_Z, launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus   = find_H(q_R, q_Z, K_R, K_zeta-delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    d2H_dKzeta2 = (H_plus - 2*H_current + H_minus) / (delta_K_zeta**2)    
    
    return d2H_dKzeta2

def find_d2H_dKzeta_dKz(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_zeta, delta_K_Z, 
                        interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus_K_zeta_plus_K_z   = find_H(q_R, q_Z, K_R, K_zeta+delta_K_zeta, K_Z+delta_K_Z, launch_angular_frequency, mode_flag, 
                                    interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_plus_K_zeta_minus_K_z  = find_H(q_R, q_Z, K_R, K_zeta+delta_K_zeta, K_Z-delta_K_Z, launch_angular_frequency, mode_flag, 
                                    interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_zeta_plus_K_z  = find_H(q_R, q_Z, K_R, K_zeta-delta_K_zeta, K_Z+delta_K_Z, launch_angular_frequency, mode_flag, 
                                    interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_zeta_minus_K_z = find_H(q_R, q_Z, K_R, K_zeta-delta_K_zeta, K_Z-delta_K_Z, launch_angular_frequency, mode_flag, 
                                    interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    d2H_dKzeta_dKz = (H_plus_K_zeta_plus_K_z - H_plus_K_zeta_minus_K_z - H_minus_K_zeta_plus_K_z + H_minus_K_zeta_minus_K_z) / (4*delta_K_zeta*delta_K_Z)    
    
    return d2H_dKzeta_dKz


def find_d2H_dKZ2(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_Z, 
                  interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus    = find_H(q_R, q_Z, K_R, K_zeta, K_Z+delta_K_Z, launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_current = find_H(q_R, q_Z, K_R, K_zeta, K_Z,           launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus   = find_H(q_R, q_Z, K_R, K_zeta, K_Z-delta_K_Z, launch_angular_frequency, mode_flag, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    d2H_dKZ2 = (H_plus - 2*H_current + H_minus) / (delta_K_Z**2)   
    
    return d2H_dKZ2


def find_d2H_dKR_dR(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_R, delta_R, 
                    interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus_K_R_plus_R   = find_H(q_R+delta_R, q_Z, K_R+delta_K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_plus_K_R_minus_R  = find_H(q_R-delta_R, q_Z, K_R+delta_K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_R_plus_R  = find_H(q_R+delta_R, q_Z, K_R-delta_K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_R_minus_R = find_H(q_R-delta_R, q_Z, K_R-delta_K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    d2H_dKR_dR = (H_plus_K_R_plus_R - H_plus_K_R_minus_R - H_minus_K_R_plus_R + H_minus_K_R_minus_R) / (4*delta_K_R*delta_R)
    
    return d2H_dKR_dR
 
    
def find_d2H_dKR_dZ(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_R, delta_Z, 
                    interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus_K_R_plus_Z   = find_H(q_R, q_Z+delta_Z, K_R+delta_K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_plus_K_R_minus_Z  = find_H(q_R, q_Z-delta_Z, K_R+delta_K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_R_plus_Z  = find_H(q_R, q_Z+delta_Z, K_R-delta_K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_R_minus_Z = find_H(q_R, q_Z-delta_Z, K_R-delta_K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    d2H_dKR_dZ = (H_plus_K_R_plus_Z - H_plus_K_R_minus_Z - H_minus_K_R_plus_Z + H_minus_K_R_minus_Z) / (4*delta_K_R*delta_Z)
    
    return d2H_dKR_dZ
  
    
def find_d2H_dKzeta_dR(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_zeta, delta_R, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus_K_zeta_plus_R   = find_H(q_R+delta_R, q_Z, K_R, K_zeta+delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                  interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_plus_K_zeta_minus_R  = find_H(q_R-delta_R, q_Z, K_R, K_zeta+delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                  interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_zeta_plus_R  = find_H(q_R+delta_R, q_Z, K_R, K_zeta-delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                  interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_zeta_minus_R = find_H(q_R-delta_R, q_Z, K_R, K_zeta-delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                  interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    d2H_dKzeta_dR = (H_plus_K_zeta_plus_R - H_plus_K_zeta_minus_R - H_minus_K_zeta_plus_R + H_minus_K_zeta_minus_R) / (4*delta_K_zeta*delta_R)  
    
    return d2H_dKzeta_dR

 
def find_d2H_dKzeta_dZ(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_zeta, delta_Z, 
                       interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus_K_zeta_plus_Z   = find_H(q_R, q_Z+delta_Z, K_R, K_zeta+delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                  interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_plus_K_zeta_minus_Z  = find_H(q_R, q_Z-delta_Z, K_R, K_zeta+delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                  interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_zeta_plus_Z  = find_H(q_R, q_Z+delta_Z, K_R, K_zeta-delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                  interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_zeta_minus_Z = find_H(q_R, q_Z-delta_Z, K_R, K_zeta-delta_K_zeta, K_Z, launch_angular_frequency, mode_flag, 
                                  interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)    
    d2H_dKzeta_dZ = (H_plus_K_zeta_plus_Z - H_plus_K_zeta_minus_Z - H_minus_K_zeta_plus_Z + H_minus_K_zeta_minus_Z) / (4*delta_K_zeta*delta_Z)  
    
    return d2H_dKzeta_dZ


def find_d2H_dKZ_dR(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_Z, delta_R, 
                    interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus_K_Z_plus_R   = find_H(q_R+delta_R, q_Z, K_R, K_zeta, K_Z+delta_K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_plus_K_Z_minus_R  = find_H(q_R-delta_R, q_Z, K_R, K_zeta, K_Z+delta_K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_Z_plus_R  = find_H(q_R+delta_R, q_Z, K_R, K_zeta, K_Z-delta_K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_Z_minus_R = find_H(q_R-delta_R, q_Z, K_R, K_zeta, K_Z-delta_K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    d2H_dKZ_dR = (H_plus_K_Z_plus_R - H_plus_K_Z_minus_R - H_minus_K_Z_plus_R + H_minus_K_Z_minus_R) / (4*delta_K_Z*delta_R)  
    
    return d2H_dKZ_dR


def find_d2H_dKZ_dZ(q_R, q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag, delta_K_Z, delta_Z, 
                    interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z):
    
    H_plus_K_Z_plus_Z   = find_H(q_R, q_Z+delta_Z, K_R, K_zeta, K_Z+delta_K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_plus_K_Z_minus_Z  = find_H(q_R, q_Z-delta_Z, K_R, K_zeta, K_Z+delta_K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_Z_plus_Z  = find_H(q_R, q_Z+delta_Z, K_R, K_zeta, K_Z-delta_K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    H_minus_K_Z_minus_Z = find_H(q_R, q_Z-delta_Z, K_R, K_zeta, K_Z-delta_K_Z, launch_angular_frequency, mode_flag, 
                                 interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
    d2H_dKZ_dZ = (H_plus_K_Z_plus_Z - H_plus_K_Z_minus_Z - H_minus_K_Z_plus_Z + H_minus_K_Z_minus_Z) / (4*delta_K_Z*delta_Z)  
    
    return d2H_dKZ_dZ


# Functions (analysis)
    # These are not strictly necessary for beam tracing, but useful for analysis of DBS
def find_dbhat_dR(q_R, q_Z, delta_R, interp_B_R, interp_B_T, interp_B_Z): 
    # \fract{d b_hat}{d R}
    B_R_plus_R = np.squeeze(interp_B_R(q_R+delta_R, q_Z))
    B_T_plus_R = np.squeeze(interp_B_T(q_R+delta_R, q_Z))
    B_Z_plus_R = np.squeeze(interp_B_Z(q_R+delta_R, q_Z))

    B_R_minus_R = np.squeeze(interp_B_R(q_R-delta_R, q_Z))
    B_T_minus_R = np.squeeze(interp_B_T(q_R-delta_R, q_Z))
    B_Z_minus_R = np.squeeze(interp_B_Z(q_R-delta_R, q_Z))
    
    B_Total_plus = np.sqrt(B_R_plus_R**2 + B_T_plus_R**2 + B_Z_plus_R**2)
    b_hat_plus = np.array([B_R_plus_R, B_T_plus_R, B_Z_plus_R]) / B_Total_plus
    
    B_Total_minus = np.sqrt(B_R_minus_R**2 + B_T_minus_R**2 + B_Z_minus_R**2)
    b_hat_minus = np.array([B_R_minus_R, B_T_minus_R, B_Z_minus_R]) / B_Total_minus    
    
    dbhat_dR = (b_hat_plus - b_hat_minus) / (2 * delta_R)
        
    return dbhat_dR

def find_dbhat_dZ(q_R, q_Z, delta_Z, interp_B_R, interp_B_T, interp_B_Z): 
    # \fract{d b_hat}{d R}
    B_R_plus_Z = np.squeeze(interp_B_R(q_R, q_Z+delta_Z))
    B_T_plus_Z = np.squeeze(interp_B_T(q_R, q_Z+delta_Z))
    B_Z_plus_Z = np.squeeze(interp_B_Z(q_R, q_Z+delta_Z))

    B_R_minus_Z = np.squeeze(interp_B_R(q_R, q_Z-delta_Z,))
    B_T_minus_Z = np.squeeze(interp_B_T(q_R, q_Z-delta_Z,))
    B_Z_minus_Z = np.squeeze(interp_B_Z(q_R, q_Z-delta_Z,))
    
    B_Total_plus = np.sqrt(B_R_plus_Z**2 + B_T_plus_Z**2 + B_Z_plus_Z**2)
    b_hat_plus = np.array([B_R_plus_Z, B_T_plus_Z, B_Z_plus_Z]) / B_Total_plus
    
    B_Total_minus = np.sqrt(B_R_minus_Z**2 + B_T_minus_Z**2 + B_Z_minus_Z**2)
    b_hat_minus = np.array([B_R_minus_Z, B_T_minus_Z, B_Z_minus_Z]) / B_Total_minus    
    
    dbhat_dZ = (b_hat_plus - b_hat_minus) / (2 * delta_Z)
        
    return dbhat_dZ

def find_waist(width, wavenumber, curvature): #Finds the size of the waist (assumes vacuum propagation and circular beam)
    waist = width / np.sqrt(1+curvature**2 * width**4 * wavenumber**2 / 4)
    return waist

def find_distance_from_waist(width, wavenumber, curvature): #Finds how far you are from the waist (assumes vacuum propagation and circular beam)
    waist = width / np.sqrt(1+curvature**2 * width**4 * wavenumber**2 / 4)
    distance_from_waist = np.sign(curvature)*np.sqrt((width**2 - waist**2)*waist**2*wavenumber**2/4)
    return distance_from_waist

#----------------------------------
