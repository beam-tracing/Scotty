"""
Created: 11/08/2022
Last Updated: 18/08/2022
@author: Tan Zheng Yang

Cartesian modification of the Scotty_beam_me_up file in the original Scotty
code by Dr Valerian Hall-Chen

============================= FUNCTION DESCRIPTION ============================

To be filled in later

===============================================================================

The below documentation 'PLAN', 'NOTES', 'COORDINATES', 'ABBREVIATIONS',
'ANGLES' and 'UNITS' are from the original Scotty_beam_me_up file

==================================== PLAN ====================================
Output everything to a file, and then do the analysis on that file.

1) Check that gradK_xi and such are done correctly, check that K_mag is 
   calculated correctly when K_zeta is nonzero
2) Check that the calculation of Psi makes sense (and the rotation angle)
3) Check that K initial's calculation makes sense


==================================== NOTES ====================================
- The loading of the input files was taken from integral_5 and modified.
- I should launch the beam inside the last closed flux surface
- K**2 = K_R**2 + K_z**2 + (K_zeta/r_R)**2, and K_zeta is constant (mode number). 
  See 14 Sep 2018 notes.

================================= COORDINATES =================================
X,Y,Z - Lab Cartesian coordinates
R,zeta,Z - Lab cylindrical coordinates
x,y,g - Beam coordinates
u1,u2,u_parallel - Field-aligned coordinates

================================ ABBREVIATIONS ================================
Abbreviations
bs - backscattered
loc - localisation
cum_loc - cumulative_localisation
ne - equilibrium electron density

=================================== ANGLES ===================================
theta - angle between g and u1, small when mismatch is small
theta_m - mismatch angle, angle between u1 and K

==================================== UNITS ====================================
- SI units
- Distance in m
- Angles in rad
- electron cyclotron frequency positive
- K normalised such that K = 1 in vacuum. (Not implemented yet)
- Distance not normalised yet, should give it thought
- Start in vacuum, otherwise Psi_3D_beam_initial_cartersian does not get done properly
"""

#============================== FUNCTION IMPORTS ==============================
# I have removed some imports that were not used in the 
# original Scotty_beam_me_up

import numpy as np
import math
from scipy import interpolate as interpolate
from scipy import integrate as integrate
from scipy import constants as constants
import matplotlib.pyplot as plt
import os
import sys
#from netCDF4 import Dataset
import bisect
import time
import platform

from Scotty_fun_general_CART import read_floats_into_list_until, find_nearest 
from Scotty_fun_general_CART import contract_special, make_unit_vector_from_cross_product
from Scotty_fun_general_CART import find_inverse_2D, find_x0, find_waist
from Scotty_fun_general_CART import find_normalised_gyro_freq
from Scotty_fun_general_CART import find_epsilon_para, find_epsilon_perp, find_epsilon_g
from Scotty_fun_general_CART import find_dbhat_dX, find_dbhat_dY, find_dbhat_dZ
from Scotty_fun_general_CART import find_d_poloidal_flux_dX, find_d_poloidal_flux_dY
from Scotty_fun_general_CART import find_d_poloidal_flux_dZ, find_Psi_3D_plasma
from Scotty_fun_general_CART import find_H_Cardano, find_D

from Scotty_fun_evolution_CART import ray_evolution_2D_fun, beam_evolution_fun


from Scotty_fun_FFD_CART import find_dH_dX, find_dH_dY, find_dH_dZ 
from Scotty_fun_CFD_CART import find_dH_dKX, find_dH_dKY, find_dH_dKZ
from Scotty_fun_FFD_CART import find_dpolflux_dX, find_dpolflux_dY, find_dpolflux_dZ
#==============================================================================

"""
DESCRIPTION
==============================

MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
==============================

INPUT
==============================

OUTPUT
==============================

"""


def beam_me_up(poloidal_launch_angle_Torbeam, # change
               toroidal_launch_angle_Torbeam, # change
               launch_freq_GHz,
               mode_flag,
               launch_beam_width,
               launch_beam_radius_of_curvature,
               launch_position, # in [X, Y, Z]
               #====================== KEYWORD ARGUMENTS ======================
               vacuumLaunch_flag                 = True,               
               find_B_method                     = 'torbeam',
               shot                              = None,
               equil_time                        = None,
               vacuum_propagation_flag           = False,
               Psi_BC_flag                       = False,
               poloidal_flux_enter               = None,
               #================== INPUT AND OUTPUT SETTINGS ==================
               ne_data_path                      = None,
               magnetic_data_path                = None,
               input_filename_suffix             = '',
               output_filename_suffix            = '',
               figure_flag                       = True,
               #=============== FOR LAUNCHING WITHIN THE PLASMA ===============
               plasmaLaunch_K                    = np.zeros(3),
               plasmaLaunch_Psi_3D_lab_Cartesian = np.zeros([3,3]),
               density_fit_parameters            = None,
               #================= FOR CIRCULAR FLUX SURFACES =================
               B_T_axis                          = None, # toroidal B-field on magnetic axis
               B_p_a                             = None, # poloidal B-field on LCFS
               R_axis                            = None, # major radius (also known as magnetic axis)
               minor_radius_a                    = None  # minor radius
               ):
    
    # NOTE: Last closed flux surface (LCFS) is the separatrix that separates the confined region
    #       of the plasma from the scrape-off layer (open B-field lines, commencing and ending
    #       on material surface)

    # Spatial grid spacing
    delta_X = 0.0001 # in the same units as data_X_coord
    delta_Y = 0.0001 # in the same units as data_Y_coord
    delta_Z = 0.0001 # in the same units as data_Z_coord
    
    # Wavevector grid spacing
    delta_K_X = 0.1 # in the same units as K_X
    delta_K_Y = 0.1 # in the same units as K_Y
    delta_K_Z = 0.1 # in the same units as K_Z
    
    print('Beam trace me up, Scotty! (Cartesian)')
    
    
    
    #============================== BASIC SETUP ===============================
    
    # calculate launch angular frequency using \omega = 2\pi f
    launch_angular_frequency = 2 * math.pi * (launch_freq_GHz * 1E9)
    
    # calculate wavenumber of launch beam using dispersion relation in 
    # vaccum \omega = ck
    wavenumber_K0 = launch_angular_frequency / constants.c
    
    # full path location of Scotty_beam_me_up_CART
    current_file_path = os.path.abspath(__file__)
    
    # current directory of Scotty_beam_me_up_CART
    current_file_fir = os.path.dirname(current_file_path)
    
    # if the input of the directory of the electron density data, ne_data_path
    # is 'None', create a data_path inside the current directory of 
    # Scotty_beam_me_up_CART
    if ne_data_path is None:
        # For windows platform
        if platform.system() == 'Windows':
            ne_data_path = os.path.dirname(os.path.abspath(__file__)) + '\\'
        # For linux platform (ignore for now)
        elif platform.system() == 'Linux':
            ne_data_path = os.path.dirname(os.path.abspath(__file__)) + '/'
            
    # if the input of the directory of the magnetic field data, magnetic_data_path
    # is 'None', create a data_path inside the current directory of 
    # Scotty_beam_me_up_CART
    if magnetic_data_path is None:
        # For windows platform
        if platform.system() == 'Windows':
            magnetic_data_path = os.path.dirname(os.path.abspath(__file__)) + '\\'
        # For linux platform (ignore for now)
        elif platform.system() == 'Linux':
            magnetic_data_path = os.path.dirname(os.path.abspath(__file__)) + '/'
            
            
            
    #================ ELECTRON DENSITY INTERPOLATION FUNCTION ================
    
    # CASE 1: Electron density fit parameters are not specified, need to
    #         perform an interpolation of experimental data in order to obtain 
    #         an interpolating function
    #
    if density_fit_parameters is None:
        print('ne(psi): loading from input file')
        
        # locate file location of the electron density data ne.dat
        ne_filename = ne_data_path + 'ne' + input_filename_suffix + '.dat'
        
        # construct array from electron density data. Separator argument is for
        # what divides different items in the data file
        ne_data = np.fromfile(ne_filename,dtype=float, sep='   ')
        
        # the second column in ne.dat is the electron density in units of 10.0**19 m-3
        ne_data_density_array = ne_data[2::2] 
        
        # the first column in ne.dat is the radial coordinate \sqrt{\psi_p}
        ne_data_radialcoord_array = ne_data[1::2]
        
        # Loading radial coord for now, makes it easier to benchmark with 
        # Torbeam. Hence, have to convert to poloidal flux
        ne_data_poloidal_flux_array = ne_data_radialcoord_array**2 
        
        # --------- not needed --------
        # My new IDL file outputs the flux density directly, instead of radialcoord       
        # ne_data_poloidal_flux_array = ne_data[1::2]  
        # -----------------------------
        
        # Create the interpolation function of the electron density as a
        # function of the poloidal flux.
        #
        # fill_value = 0 as the density is 0 outside the LCFS
        # 
        # For the choice of splines, use 'linear' instead of 'cubic' if the 
        # density data has a discontinuity in the first derivative 
        interp_density_1D = interpolate.interp1d(ne_data_poloidal_flux_array, 
                                                 ne_data_density_array,
                                                 kind='cubic', axis=-1, 
                                                 copy=True, bounds_error=False,
                                                 fill_value=0, 
                                                 assume_sorted=False) 
        
        # obtain electron density from poloidal flux using the interpolation
        # function above
        def find_density_1D(poloidal_flux, 
                            interp_density_1D = interp_density_1D):
            """
            DESCRIPTION
            ==============================
            Finds the electron density given the poloidal flux using the
            interpolation function above.

            INPUT
            ==============================
            poloidal_flux (float):
                Value of the poloidal flux to be interpolated.
                
            interp_density_1D (function):
                The electron density interpolation function. By default, it is
                the interpolation function we have constructed above.

            OUTPUT
            ==============================
            density (float):
                The electron density corresponding to the given poloidal flux
                obtained from the interpolation function.
            """
            density = interp_density_1D(poloidal_flux)
            return density
        
    # CASE 2: 4 electron fit parameters A, B, C, D are given to construct 
    #         an electron density function that is of the form:
    #               electron_density = (Ax + B) * tanh(Cx + D)
    #         where x is the poloidal flux   
    #
    elif len(density_fit_parameters) == 4:
            
        print('ne(psi): Using order_1_polynomial * tanh(order_1_polynomial)')
        
        # So that saving the input data later does not complain 
        ne_data_density_array=None 
        ne_data_radialcoord_array=None 
        
        def find_density_1D(poloidal_flux, 
                            poloidal_flux_enter = poloidal_flux_enter,
                            density_fit_parameters=density_fit_parameters):
            """
            DESCRIPTION
            ==============================
            Finds the electron density given the poloidal flux by assuming that:
                electron_density = (Ax + B) * tanh(Cx + D)
            where x is the poloidal flux

            INPUT
            ==============================
            poloidal_flux (float):
                Value of the poloidal flux in which we would like to find the
                corresponding electron density.
                
            poloidal_flux_enter (float):
                Value of the poloidal flux entering the plasma.
                
            density_fit_parameters (numpy array):
                contains the values A, B, C and D where
                    A = density_fit_parameters[0]
                    B = density_fit_parameters[1]
                    C = density_fit_parameters[2]
                    D = density_fit_parameters[3]

            OUTPUT
            ==============================
            density (float):
                The electron density corresponding to the given poloidal flux.
            """
            
            # For code readability
            A = density_fit_parameters[0]
            B = density_fit_parameters[1]
            C = density_fit_parameters[2]
            D = density_fit_parameters[3]
            
            density_fit = (
                            (A * poloidal_flux + B) * 
                            np.tanh(C * poloidal_flux + D)
                            )
            
            # Setup Boolean array to set entries of the poloidal flux that is
            # outside poloidal_flux_enter to zero. Exploits the fact that in
            # Python, False == 0, True == 1
            is_inside = poloidal_flux <= poloidal_flux_enter 
            density = is_inside * density_fit 
            
            return density     
        
    # CASE 3: 3 electron fit parameters A, B, C are given to construct 
    #         an electron density function that is of the form:
    #               electron_density = A * tanh(Bx + D)
    #         where x is the poloidal flux   
    #
    elif len(density_fit_parameters) == 3:
        
        print('ne(psi): using constant * tanh(order_1_polynomial)')
        
        # So that saving the input data later does not complain 
        ne_data_density_array = None 
        ne_data_radialcoord_array = None 
        
        def find_density_1D(poloidal_flux, 
                            poloidal_flux_enter=poloidal_flux_enter,
                            density_fit_parameters=density_fit_parameters):
            """
            DESCRIPTION
            ==============================
            Finds the electron density given the poloidal flux by assuming that:
                electron_density = A * tanh(Bx + C)
            where x is the poloidal flux

            INPUT
            ==============================
            poloidal_flux (float):
                Value of the poloidal flux in which we would like to find the
                corresponding electron density.
                
            poloidal_flux_enter (float):
                Value of the poloidal flux entering the plasma.
                
            density_fit_parameters (numpy array):
                contains the values A, B, and C where
                    A = density_fit_parameters[0]
                    B = density_fit_parameters[1]
                    C = density_fit_parameters[2]

            OUTPUT
            ==============================
            density (float):
                The electron density corresponding to the given poloidal flux.
            """
            
            # For code readability
            A = density_fit_parameters[0]
            B = density_fit_parameters[1]
            C = density_fit_parameters[2]
            
            density_fit = A * np.tanh(B * poloidal_flux + C)
            
            # Setup Boolean array to set entries of the poloidal flux that is
            # outside poloidal_flux_enter to zero. Exploits the fact that in
            # Python, False == 0 and True == 1
            is_inside = poloidal_flux <= poloidal_flux_enter 
            density = is_inside * density_fit 

            return density
    
    # CASE 4: 2 electron fit parameters A, B are given to construct 
    #         an electron density function that is of the form:
    #               electron_density = A * (1 - x^2/B)
    #         where x is the poloidal flux. In other words, we are using a
    #         quadratic fit
    #
    #         Note that A is the density at the magnetic axis (psi = 0)
    #         where A = density_fit_parameters[0]; B is the poloidal flux
    #         when density = 0 and B = density_fit_parameters[1]
    #
    # VALERIAN'S TEST CASE: Let A = 2 (y-intercept) and B = 1 (x-intercept)
    #                       such that density_fit =  A * (1 - (poloidal_flux)**2)
    # 
    #
    # NOTE THAT WE SHOULD USE THIS TO BENCH MARK CARTESIAN CODE
    elif len(density_fit_parameters) == 2: 
    
        print('ne(psi): using quadratic profile')
        
        # So that saving the input data later does not complain
        ne_data_density_array = None 
        ne_data_radialcoord_array = None
        
        # B = density_fit_parameters[1] is the poloidal flux when density = 0.
        # Therefore, it has to be equal to the initial poloidal flux entering.
        if poloidal_flux_enter != density_fit_parameters[1]:
            print('Invalid fit parameters for quadratic density profile')
        
        def find_density_1D(poloidal_flux, 
                            poloidal_flux_enter = poloidal_flux_enter,
                            density_fit_parameters=density_fit_parameters):
            
            """
            DESCRIPTION
            ==============================
            Finds the electron density given the poloidal flux by assuming that:
                electron_density = A * (1 - x**2/B)
            where x is the poloidal flux

            INPUT
            ==============================
            poloidal_flux (float):
                Value of the poloidal flux in which we would like to find the
                corresponding electron density.
                
            poloidal_flux_enter (float):
                Value of the poloidal flux entering the plasma.
                
            density_fit_parameters (numpy array):
                contains the values A, B where
                    A = density_fit_parameters[0]
                    B = density_fit_parameters[1]

            OUTPUT
            ==============================
            density (float):
                The electron density corresponding to the given poloidal flux.
            """
            
            # For code readability
            A = density_fit_parameters[0]
            B = density_fit_parameters[1]
            
            density_fit = A * (1 - poloidal_flux**2/B)
            
            # Setup Boolean array to set entries of the poloidal flux that is
            # outside poloidal_flux_enter to zero. Exploits the fact that in
            # Python, False == 0 and True == 1
            is_inside = poloidal_flux <= poloidal_flux_enter 
            density = is_inside * density_fit 

            return density
    
    # There should be no other forms of the fits of the electron density
    # other than case 1, 2, 3 and 4
    else:
        print('density_fit_parameters has an invalid length')
        sys.exit()
    
    
    
    #======================= MAGNETIC FIELD COMPONENTS ========================
    #
    # Note that the data output we have from TORBEAM, EFITT etc. is always in
    # cylindrical coordinates, therefore we need to convert to cartesian
    # manually
    
    # For the 2D interpolation functions. For no smoothing, set to 0
    interp_order = 5
    interp_smoothing = 2 
    
    # CASE 1: Use TORBEAM to calculate the magnetic field and poloidal flux
    if find_B_method == 'torbeam':   
        
        print('Using Torbeam input files for B and poloidal flux')
        
        # location of 'topfile' (data file of torbeam)
        topfile_filename = magnetic_data_path + 'topfile' + input_filename_suffix
        
        # Read data from TORBEAM (in cylindrical coordinates).
        # Start reading only from X-coords onwards
        with open(topfile_filename) as f:
            while not 'X-coordinates' in f.readline(): pass 
            data_R_coord = read_floats_into_list_until('Z-coordinates', f)
            data_Z_coord = read_floats_into_list_until('B_R', f)
            data_B_R_grid = read_floats_into_list_until('B_t', f)
            data_B_T_grid = read_floats_into_list_until('B_Z', f)
            data_B_Z_grid = read_floats_into_list_until('psi', f)
            poloidalFlux_grid = read_floats_into_list_until('you fall asleep', f)
            
        # Converts some lists to arrays so that stuff later doesn't complain
        data_R_coord = np.array(data_R_coord)
        data_Z_coord = np.array(data_Z_coord)
        
        # Row-major and column-major business (Torbeam is in Fortran and 
        # Scotty is in Python)
        data_B_R_grid = (
                        np.transpose(
                                        (
                                            np.asarray(data_B_R_grid)
                                        ).reshape(
                                                    len(data_Z_coord),
                                                    len(data_R_coord), 
                                                    order='C'
                                                    )
                                    )
                            )
        
        data_B_T_grid = (
                        np.transpose(
                                        (
                                            np.asarray(data_B_T_grid)
                                        ).reshape(len(data_Z_coord),
                                                  len(data_R_coord), 
                                                  order='C')
                                    )
                        )
                                                  
        data_B_Z_grid = (
                        np.transpose(
                                        (
                                            np.asarray(data_B_Z_grid)
                                        ).reshape(len(data_Z_coord),
                                                  len(data_R_coord), 
                                                  order='C')
                                        )
                        )
                                                      
                                                      
        poloidalFlux_grid = (
                            np.transpose(
                                            (
                                                np.asarray(poloidalFlux_grid)
                                            ).reshape(len(data_Z_coord),
                                                      len(data_R_coord), 
                                                      order='C')
                                            )
                            )
                                                      
        # Find magnetic field components (cylindrical) via interpolation
        # MIGHT NEED VALERIAN TO EXPLAIN
        interp_B_R = interpolate.RectBivariateSpline(data_R_coord,
                                                     data_Z_coord,
                                                     data_B_R_grid, 
                                                     bbox=[None, None, None, None], 
                                                     kx=interp_order, 
                                                     ky=interp_order, 
                                                     s=interp_smoothing
                                                     )
        
        interp_B_T = interpolate.RectBivariateSpline(data_R_coord,
                                                     data_Z_coord,
                                                     data_B_T_grid, 
                                                     bbox=[None, None, None, None], 
                                                     kx=interp_order, 
                                                     ky=interp_order, 
                                                     s=interp_smoothing)
        
        interp_B_Z = interpolate.RectBivariateSpline(data_R_coord,
                                                     data_Z_coord,
                                                     data_B_Z_grid, 
                                                     bbox=[None, None, None, None], 
                                                     kx=interp_order, 
                                                     ky=interp_order, 
                                                     s=interp_smoothing)
        
        def find_B_R(q_X, q_Y, q_Z):
            q_R = np.sqrt(q_X**2 + q_Y**2)
            B_R = interp_B_R(q_R, q_Z, grid=False)
            return B_R
    
        def find_B_T(q_X, q_Y, q_Z):
            q_R = np.sqrt(q_X**2 + q_Y**2)
            B_T = interp_B_T(q_R, q_Z, grid=False)
            return B_T
        
        def find_B_Z(q_X, q_Y, q_Z):
            q_R = np.sqrt(q_X**2 + q_Y**2)
            B_Z = interp_B_Z(q_R, q_Z, grid=False)
            return B_Z
        
        # Find X-component of the magnetic field 
        def find_B_X(q_X, q_Y, q_Z):
            
            # need to use arctan2 to choose the correct quadrant
            zeta = np.arctan2(q_Y, q_X)
            
            B_R = find_B_R(q_X, q_Y, q_Z)
            B_T = find_B_T(q_X, q_Y, q_Z)
            
            B_magnitude_XY_plane = np.sqrt(B_R**2 + B_T**2)
            return B_magnitude_XY_plane * np.cos(zeta)
        
        # Find Y-component of the magnetic field 
        def find_B_Y(q_X, q_Y, q_Z):
            
            # need to use arctan2 to choose the correct quadrant
            zeta = np.arctan2(q_Y, q_X)
            
            B_R = find_B_R(q_X, q_Y, q_Z)
            B_T = find_B_T(q_X, q_Y, q_Z)
            
            B_magnitude_XY_plane = np.sqrt(B_R**2 + B_T**2)
            return B_magnitude_XY_plane * np.sin(zeta)
        
        # Find poloidal flux based on interpolation in cylindrical coordinates
        interp_poloidal_flux = interpolate.RectBivariateSpline(data_R_coord,
                                                               data_Z_coord,
                                                               poloidalFlux_grid, 
                                                               bbox=[None, None, None, None], 
                                                               kx=interp_order, 
                                                               ky=interp_order, 
                                                               s=interp_smoothing)
        
        
        # Find poloidal flux in cartesian coordinates
        def interp_poloidal_flux(q_X, q_Y, q_Z, 
                                 data_R_coord, data_Z_coord, poloidalFlux_grid,
                                 interp_order, interp_smoothing):
            
            # Since the data is in cylindrical, the interpolation function is also in cylindrical
            interp_poloidal_flux = interpolate.RectBivariateSpline(data_R_coord,
                                                                   data_Z_coord,
                                                                   poloidalFlux_grid, 
                                                                   bbox=[None, None, None, None], 
                                                                   kx=interp_order, 
                                                                   ky=interp_order, 
                                                                   s=interp_smoothing)
            
            q_R = np.sqrt(q_X**2 + q_Y**2)
            
            polflux = interp_poloidal_flux.ev(q_R, q_Z)
            
            return polflux
        
        # To prevent the data-saving routines from complaining later on
        efit_time = None
    
    # CASE 2: Compute the magnetic field and the poloidal flux based on an
    #         analytical equilibrium in Valerian's paper. See equations
    #         (80), (81), (82) and (83)
    elif find_B_method == 'analytical':
        
        # ----------------- VALERIAN NOTES -----------------
        # B_p (hence B_R and B_Z) physical when inside the LCFS, have not 
        # implemented calculation outside the LCFS
        # To do that, need to make sure B_p = B_p_a outside the LCFS
        # --------------------------------------------------
        
        print('Using analytical input for B and poloidal flux')
        
        # Toroidal component of the magnetic field. See equation (80) of Valerian's paper
        def find_B_T(q_X, q_Y, q_Z, 
                     R_axis = R_axis, B_T_axis = B_T_axis):
            
            q_R = np.sqrt(q_X**2 + q_Y**2)
            B_T = B_T_axis * (R_axis/q_R)
            return B_T
        
        # Poloidal magnetic field. See equation (83) of Valerian's paper
        def find_B_p(q_X, q_Y, q_Z, R_axis, minor_radius_a, B_p_a):
            
            q_R = np.sqrt(q_X**2 + q_Y**2)
            B_p = B_p_a * (np.sqrt((q_R-R_axis)**2 + q_Z**2) / minor_radius_a)
            return B_p
        
        # R-component of the magnetic field. See equation (81) of Valerian's paper
        def find_B_R(q_X, q_Y, q_Z, 
                     R_axis = R_axis, minor_radius_a = minor_radius_a, 
                     B_p_a = B_p_a):
            
            q_R = np.sqrt(q_X**2 + q_Y**2)
            B_p = find_B_p(q_X, q_Y, q_Z, R_axis, minor_radius_a, B_p_a)
            B_R = B_p * (q_Z / np.sqrt((q_R-R_axis)**2 + q_Z**2))
            return B_R
    
        # Z-component of the magnetic field. See equation (82) of Valerian's paper
        def find_B_Z(q_X, q_Y, q_Z,
                     R_axis = R_axis, minor_radius_a = minor_radius_a, 
                     B_p_a = B_p_a):
            
            q_R = np.sqrt(q_X**2 + q_Y**2)
            B_p = find_B_p(q_X, q_Y, q_Z, R_axis, minor_radius_a, B_p_a)
            B_Z = B_p * ((q_R-R_axis) / np.sqrt((q_R-R_axis)**2 + q_Z**2))
            return B_Z
        
        # X-component of the magnetic field
        def find_B_X(q_X, q_Y, q_Z,
                     R_axis = R_axis, minor_radius_a = minor_radius_a, 
                     B_p_a = B_p_a):
            
            # need to use arctan2 to choose the quadrant correctly
            zeta = np.arctan2(q_Y, q_X)
            
            B_R = find_B_R(q_X, q_Y, q_Z, 
                         R_axis = R_axis, minor_radius_a = minor_radius_a, 
                         B_p_a = B_p_a)
            
            B_T = find_B_T(q_X, q_Y, q_Z, 
                         R_axis = R_axis, B_T_axis = B_T_axis)
            
            B_magnitude_XY_plane = np.sqrt(B_R**2 + B_T**2)
            
            return B_magnitude_XY_plane * np.cos(zeta)
        
        # Y-component of the magnetic field
        def find_B_Y(q_X, q_Y, q_Z,
                     R_axis = R_axis, minor_radius_a = minor_radius_a, 
                     B_p_a = B_p_a):
            
            # need to use arctan2 to choose the quadrant correctly
            zeta = np.arctan2(q_Y, q_X)
            
            B_R = find_B_R(q_X, q_Y, q_Z, 
                         R_axis = R_axis, minor_radius_a = minor_radius_a, 
                         B_p_a = B_p_a)
            
            B_T = find_B_T(q_X, q_Y, q_Z, 
                         R_axis = R_axis, B_T_axis = B_T_axis)
            
            B_magnitude_XY_plane = np.sqrt(B_R**2 + B_T**2)
            
            return B_magnitude_XY_plane * np.sin(zeta)
        
        # Compute poloidal flux
        def interp_poloidal_flux(q_X, q_Y, q_Z,
                                 R_axis = R_axis,minor_radius_a = minor_radius_a,
                                 grid=None):
            # keyword 'grid' doesn't do anything; it's a fix to prevent
            # other routines from complaining
            q_R = np.sqrt(q_X**2 + q_Y**2)
            polflux = np.sqrt((q_R-R_axis)**2 + (q_Z)**2) / minor_radius_a
            return polflux
        
        # I DON'T KNOW IF I NEED THIS
        #
        # Not strictly necessary, but helpful for visualisation
        data_R_coord = np.linspace(R_axis-minor_radius_a, R_axis+minor_radius_a,101)
        data_Z_coord = np.linspace(-minor_radius_a, minor_radius_a,101)
        poloidalFlux_grid = interp_poloidal_flux(*np.meshgrid(data_R_coord, data_Z_coord, sparse=False, indexing='ij'))
        
    elif (find_B_method == 'EFITpp') or (find_B_method == 'UDA_saved'):
        
        if find_B_method == 'EFITpp':
            
            print('Using MSE-constrained EFIT++ output files directly for B and poloidal flux')
            
            # CHECK WITH VALERIAN, NOT SURE HOW THE DATA LOOKS LIKE
            
            
        elif find_B_method == 'UDA_saved' and shot <= 30471: # MAST
            
            # 30471 is the last shot on MAST
            # data saved differently for MAST-U shots
            print(shot)
            
            
            # CHECK WITH VALERIAN, NOT SURE HOW THE DATA LOOKS LIKE
            
        elif find_B_method == 'UDA_saved' and shot > 30471: # MAST-U
        
            # 30471 is the last shot on MAST
            # data saved differently for MAST-U shots
            
            # CHECK WITH VALERIAN, NOT SURE HOW THE DATA LOOKS LIKE
            
            print('test')
            
    elif find_B_method == 'curvy_slab':
            
        print('Analytical curvy slab geometry')
        
        def find_B_R(q_X, q_Y, q_Z):
            q_R = np.sqrt(q_X**2 + q_Y**2)
            return np.zeros_like(q_R)
        
        def find_B_T(q_X, q_Y, q_Z,
                     B_T_axis = B_T_axis, R_axis = R_axis):
            
            q_R = np.sqrt(q_X**2 + q_Y**2)
            B_T = B_T_axis * R_axis / q_R
            return B_T
        
        def find_B_Z(q_X, q_Y, q_Z):
            q_R = np.sqrt(q_X**2 + q_Y**2)
            return np.zeros_like(q_R)
        
        # Find cartesian components
        def find_B_X(q_X, q_Y, q_Z,
                     B_T_axis = B_T_axis, R_axis = R_axis):
            
            zeta = np.arctan2(q_Y, q_X)
            
            B_R = find_B_R(q_X, q_Y, q_Z)
            B_T = find_B_T(q_X, q_Y, q_Z,
                         B_T_axis = B_T_axis, R_axis = R_axis)
            
            B_magnitude_XY = np.sqrt(B_R**2 + B_T**2)
            
            return B_magnitude_XY * np.cos(zeta)
        
        def find_B_Y(q_X, q_Y, q_Z,
                     B_T_axis = B_T_axis, R_axis = R_axis):
            
            zeta = np.arctan2(q_Y, q_X)
            
            B_R = find_B_R(q_X, q_Y, q_Z)
            B_T = find_B_T(q_X, q_Y, q_Z,
                         B_T_axis = B_T_axis, R_axis = R_axis)
            
            B_magnitude_XY = np.sqrt(B_R**2 + B_T**2)
            
            return B_magnitude_XY * np.sin(zeta)


    elif find_B_method == 'test':
        # TODO: tidy up
        # Works nicely with the new MAST-U UDA output
        
        # ASK VALERIAN WHAT THIS TEST CASE IS ABOUT
        print('test')        
    
    else:
        print('Invalid find_B_method')
        sys.exit()
    
    
    
    
    #============================= LAUNCH PARAMETERS ==========================
    
    # CASE 1: Probe beam launched outside plasma
    if vacuumLaunch_flag: # remember that True == 1 and False == 0
    
        print('Beam launched from outside the plasma')
        
        # Equation (78) of Valerian's paper. Used to initialize K_ant in
        # cylindrical coordinates
        toroidal_launch_angle_Torbeam_RADIANS = toroidal_launch_angle_Torbeam/180.0*math.pi
        poloidal_launch_angle_Torbeam_RADIANS = poloidal_launch_angle_Torbeam/180.0*math.pi 
        
        K_X_launch    = (-wavenumber_K0 
                         * np.cos( toroidal_launch_angle_Torbeam_RADIANS) 
                         * np.cos( poloidal_launch_angle_Torbeam_RADIANS) 
                         )
        K_Y_launch = (-wavenumber_K0
                         * np.sin( toroidal_launch_angle_Torbeam_RADIANS ) 
                         * np.cos( poloidal_launch_angle_Torbeam_RADIANS) 
                         )
        K_Z_launch    = (-wavenumber_K0 
                         * np.sin( poloidal_launch_angle_Torbeam_RADIANS ))
        
        launch_K = np.array([K_X_launch, K_Y_launch, K_Z_launch])
        
        # NEED EXPLANATION
        # poloidal_launch_angle_Rz = (180.0+poloidal_launch_angle_Torbeam)/180.0*np.pi
        poloidal_rotation_angle = (90.0+poloidal_launch_angle_Torbeam)/180.0*np.pi
        # toroidal_launch_angle_Rz = (180.0+toroidal_launch_angle_Torbeam)/180.0*np.pi 
        
        # Equations (15) and (16) of Valerian's paper
        #
        # Note that K = K_g at launch 
        #
        # I thought real and imaginary parts not simultaneously diagonalizable?
        Psi_w_beam_launch_cartersian = np.array(
                [
                    [ wavenumber_K0/launch_beam_radius_of_curvature + 2j*launch_beam_width**(-2), 0],
                    [ 0, wavenumber_K0/launch_beam_radius_of_curvature+2j*launch_beam_width**(-2)]
                ]
            )
        
        identity_matrix_2D = np.eye(2, 2)
        
        # Convert from beam frame to lab frame
        #
        # ASK VALERIAN ABOUT THE LOGIC
        rotation_matrix_pol = np.array( [
                [ np.cos(poloidal_rotation_angle), 0, np.sin(poloidal_rotation_angle) ],
                [ 0, 1, 0 ],
                [ -np.sin(poloidal_rotation_angle), 0, np.cos(poloidal_rotation_angle) ]
                ] )
        
        rotation_matrix_tor = np.array( [
                [ np.cos(toroidal_launch_angle_Torbeam/180.0*math.pi), np.sin(toroidal_launch_angle_Torbeam/180.0*math.pi), 0 ],
                [ -np.sin(toroidal_launch_angle_Torbeam/180.0*math.pi), np.cos(toroidal_launch_angle_Torbeam/180.0*math.pi), 0 ],
                [ 0,0,1 ]
                ] )
        
        rotation_matrix         = np.matmul(rotation_matrix_pol,rotation_matrix_tor)
        rotation_matrix_inverse = np.transpose(rotation_matrix)
        
        Psi_3D_beam_launch_cartersian = np.array([
                [ Psi_w_beam_launch_cartersian[0][0], Psi_w_beam_launch_cartersian[0][1], 0 ],
                [ Psi_w_beam_launch_cartersian[1][0], Psi_w_beam_launch_cartersian[1][1], 0 ],
                [ 0, 0, 0 ]
                ])
        
        Psi_3D_lab_launch_cartersian = np.matmul( rotation_matrix_inverse, np.matmul(Psi_3D_beam_launch_cartersian, rotation_matrix) )
                                 
        if vacuum_propagation_flag:
            
            Psi_w_beam_inverse_launch_cartersian = find_inverse_2D(Psi_w_beam_launch_cartersian)
            
            ### LOCATES THE ENTRY POINT OF THE BEAM INTO THE PLASMA ###
            #
            # In cylindrical coordinates
            # NEED TO CHECK
            R_launch = np.sqrt(launch_position[0]**2 + launch_position[1]**2)
            Z_launch = launch_position[2]
            
            search_Z_end = Z_launch - R_launch * np.tan(np.radians(poloidal_launch_angle_Torbeam))
            numberOfCoarseSearchPoints = 50
            R_coarse_search_array = np.linspace(R_launch, 0, numberOfCoarseSearchPoints)
            Z_coarse_search_array = np.linspace(Z_launch, search_Z_end, numberOfCoarseSearchPoints)
            poloidal_flux_coarse_search_array = np.zeros(numberOfCoarseSearchPoints)
            
            # need change to cartesian?
            poloidal_flux_coarse_search_array = interp_poloidal_flux(R_coarse_search_array,
                                                                     Z_coarse_search_array,
                                                                     grid=False)
            meets_flux_condition_array = (poloidal_flux_coarse_search_array 
                                          < 0.9 * poloidal_flux_enter)
            
            # Entry point found
            
            
    
    
        



















