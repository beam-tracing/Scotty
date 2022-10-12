"""
Created: 11/08/2022
Last Updated: 08/09/2022
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
- Start in vacuum, otherwise Psi_3D_beam_initial_cartesian does not get done properly
"""

#==============================================================================
#============================== FUNCTION IMPORTS ==============================
#==============================================================================

#------------------------ ORIGINAL SCOTTY IMPORTS -----------------------------
# From the OG Scotty code by Valerian
#
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

#-------------------------------- THE NEW STUFF -------------------------------
#
# add some of my own code

from Scotty_find_entry_point_CART import find_entry_point

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

# The original Scotty code launches from q_zeta = 0 meaning that Y = 0. In this 
# cartesian version however, we allow for arbitrary Y.

def beam_me_up(poloidal_launch_angle_Torbeam, # positive angle measured anti-clockwise from the negative x-axis in XZ plane
               toroidal_launch_angle_Torbeam, # positive angle measured anti-clockwise from the negative x-axis in XY plane
               launch_freq_GHz,
               mode_flag,
               launch_beam_width,
               launch_beam_curvature, # don't use radius of curvature as in the original Scotty. Avoids the issue of division by 0
               launch_position, # in [X, Y, Z]
               #---------------------- KEYWORD ARGUMENTS ----------------------
               vacuumLaunch_flag                 = True,               
               find_B_method                     = 'torbeam',
               shot                              = None,
               equil_time                        = None,
               vacuum_propagation_flag           = False, # turns analytical vacuum propagation on/off
               Psi_BC_flag                       = False,
               poloidal_flux_enter               = None,
               #------------------ INPUT AND OUTPUT SETTINGS ------------------
               ne_data_path                      = None,
               magnetic_data_path                = None,
               input_filename_suffix             = '',
               output_filename_suffix            = '',
               figure_flag                       = True,
               #--------------- FOR LAUNCHING WITHIN THE PLASMA ---------------
               plasmaLaunch_K                    = np.zeros(3),
               plasmaLaunch_Psi_3D_lab           = np.zeros([3,3]),
               density_fit_parameters            = None,
               #----------------- FOR CIRCULAR FLUX SURFACES ------------------
               B_T_axis                          = None, # toroidal B-field on magnetic axis
               B_p_a                             = None, # poloidal B-field on LCFS
               R_axis                            = None, # major radius (also known as magnetic axis)
               minor_radius_a                    = None  # minor radius
               ):
    
    # NOTE: Last closed flux surface (LCFS) is the separatrix that separates the confined region
    #       of the plasma from the scrape-off layer (open B-field lines, commencing and ending
    #       on material surface)

    # Spatial grid spacing
    delta_X = -0.0001 # in the same units as data_X_coord
    delta_Y = 0.0001 # in the same units as data_Y_coord
    delta_Z = 0.0001 # in the same units as data_Z_coord
    
    # Wavevector grid spacing
    delta_K_X = 0.1 # in the same units as K_X
    delta_K_Y = 0.1 # in the same units as K_Y
    delta_K_Z = 0.1 # in the same units as K_Z
    
    print('Beam trace me up, Scotty! (Cartesian)')
    
    
    
    #==========================================================================
    #                               BASIC SETUP 
    #==========================================================================
    
    # calculate launch angular frequency using \omega = 2\pi f
    launch_angular_frequency = 2 * math.pi * (launch_freq_GHz * 1E9)
    
    # calculate wavenumber of launch beam using dispersion relation in 
    # vaccum \omega = ck
    wavenumber_K0 = launch_angular_frequency / constants.c
    
    # full path location of Scotty_beam_me_up_CART
    current_file_path = os.path.abspath(__file__)
    
    # current directory of Scotty_beam_me_up_CART
    current_file_dir = os.path.dirname(current_file_path)
    
    # if the input of the directory of the electron density data, ne_data_path
    # is 'None', create a data_path inside the current directory of 
    # Scotty_beam_me_up_CART
    if ne_data_path is None:
        # For windows platform
        if platform.system() == 'Windows':
            ne_data_path = current_file_dir + '\\'
        # For linux platform (ignore for now)
        elif platform.system() == 'Linux':
            ne_data_path = current_file_dir + '/'
            
    # if the input of the directory of the magnetic field data, magnetic_data_path
    # is 'None', create a data_path inside the current directory of 
    # Scotty_beam_me_up_CART
    if magnetic_data_path is None:
        # For windows platform
        if platform.system() == 'Windows':
            magnetic_data_path = current_file_dir + '\\'
        # For linux platform (ignore for now)
        elif platform.system() == 'Linux':
            magnetic_data_path = current_file_dir + '/'
            
            
            
    #==========================================================================
    #                 ELECTRON DENSITY INTERPOLATION FUNCTION 
    #==========================================================================
    
    # So that saving the input data later does not complain
    ne_data_density_array = None 
    ne_data_radialcoord_array = None
    
    def find_density_1D(poloidal_flux):
        """
        DESCRIPTION
        ==============================
        Finds the electron density given the poloidal flux. We assume a simple
        case of a linear relation between the density and the poloidal flux.
        
        We let the y-intercept be n_e = 1 and the x-intercept to be equal to
        poloidal_flux = poloidal_flux_enter

        INPUT
        ==============================
        poloidal_flux (float):
            Value of the poloidal flux to be interpolated.

        OUTPUT
        ==============================
        density (float):
            The electron density corresponding to the given poloidal flux.
        """
        if poloidal_flux > poloidal_flux_enter:
            density = 0
        else:
            density = -5 * poloidal_flux + 5
        
        return density
    
    
        
    #==========================================================================
    #                        MAGNETIC FIELD COMPONENTS 
    #==========================================================================
    
    # Define this later to read B_X, B_Y and B_Z from a pre-processed file
    #
    # also need to define interp_poloidal_flux(x, y, z). Probably need to use
    # RegularGridInterpolator from scipy
    #
    # output data normally in RZ coordinates
    
    # CASE 1: Magnetic field components are obtained through interpolation from
    #         EFIT data output
    #
    if find_B_method == 'EFIT':
    
        data_file_name = magnetic_data_path + 'test_data\\' + str(shot) + '_equilibrium_data_CART.npz'
        
        loadfile                     = np.load(data_file_name)
        data_X_coord                 = loadfile['data_X_coord']
        data_Y_coord                 = loadfile['data_Y_coord']
        data_Z_coord                 = loadfile['data_Z_coord']
        poloidalFlux_grid_all_times  = loadfile['poloidalFlux_grid_all_times']
        Bx_grid_all_times            = loadfile['Bx_grid_all_times']
        By_grid_all_times            = loadfile['By_grid_all_times']
        Bz_grid_all_times            = loadfile['Bz_grid_all_times']
        time_EFIT                    = loadfile['time_EFIT']
        
        #-------------- INTERPOLATING FUNCTION FOR POLOIDAL FLUX ------------------
        
        t_idx = find_nearest(time_EFIT, equil_time)
        print('EFIT time', time_EFIT[t_idx])
        
        # find the poloidal flux and B-field component grids for the equilibrium time
        poloidalFlux_grid = poloidalFlux_grid_all_times[t_idx,:,:,:] 
        Bx_grid = Bx_grid_all_times[t_idx,:,:,:] 
        By_grid = By_grid_all_times[t_idx,:,:,:] 
        Bz_grid = Bz_grid_all_times[t_idx,:,:,:] 
        
        interp_poloidal_flux = interpolate.RegularGridInterpolator(
                                                                        (data_X_coord,
                                                                        data_Y_coord,
                                                                        data_Z_coord
                                                                        ),  
                                                                    poloidalFlux_grid,
                                                                    method = "quintic",
                                                                    bounds_error = False,
                                                                    fill_value = None
                                                                    )
        
        interp_Bx = interpolate.RegularGridInterpolator(
                                                             (data_X_coord,
                                                             data_Y_coord,
                                                             data_Z_coord
                                                             ),  
                                                        Bx_grid,
                                                        method = "quintic",
                                                        bounds_error = False,
                                                        fill_value = None
                                                        )
        
        interp_By = interpolate.RegularGridInterpolator(
                                                             (data_X_coord,
                                                             data_Y_coord,
                                                             data_Z_coord
                                                             ),  
                                                        By_grid,
                                                        method = "quintic",
                                                        bounds_error = False,
                                                        fill_value = None
                                                        )
        
        interp_Bz = interpolate.RegularGridInterpolator(
                                                             (data_X_coord,
                                                             data_Y_coord,
                                                             data_Z_coord
                                                             ),  
                                                        Bz_grid,
                                                        method = "quintic",
                                                        bounds_error = False,
                                                        fill_value = None
                                                        )
        def find_poloidal_flux(q_X, q_Y, q_Z):
            return interp_poloidal_flux((q_X, q_Y, q_Z))
        
        def find_B_X(q_X, q_Y, q_Z):
            return interp_Bx((q_X, q_Y, q_Z))
        
        def find_B_Y(q_X, q_Y, q_Z):
            return interp_By((q_X, q_Y, q_Z))
        
        def find_B_Z(q_X, q_Y, q_Z):
            return interp_Bz((q_X, q_Y, q_Z))
        

    
    # CASE 2: Magnetic field are components are obtained by considering a
    #         slab geometry
    elif find_B_method == 'Slab':
        
        def find_poloidal_flux(q_X, q_Y, q_Z):
            '''
            In a slab geometry, 
            '''
            return 0.5*q_X
        
        def find_B_X(q_X, q_Y, q_Z):
            return 0.0
        
        def find_B_Y(q_X, q_Y, q_Z):
            return 1.0
        
        def find_B_Z(q_X, q_Y, q_Z):
            return 0.0
        
    
    
    
   
    
    
    #==========================================================================
    #                              LAUNCH PARAMETERS 
    #==========================================================================
    
    #-------------------------------------------------------------------------
    # CASE 1: Probe beam launched outside plasma (in vacuum)
    #-------------------------------------------------------------------------
    if vacuumLaunch_flag: # remember that True == 1 and False == 0
    
        print('Beam launched from outside the plasma')
        
        #------------------------ INITIALIZE K VECTOR -------------------------
        #
        # NOTE: WE SHOULD MODIFY THIS PART OF THE CODE TO ALLOW FOR MORE CASES
        #       OTHER THAN q_zeta = 0!!!
        #
        # Equation (78) of Valerian's paper 
        #
        # In equation (56), for q_zeta = 0, we have K_X = K_R
        K_X_launch    = (-wavenumber_K0 
                         * np.cos(np.radians(toroidal_launch_angle_Torbeam))
                         * np.cos(np.radians(poloidal_launch_angle_Torbeam))
                         )
        
        # In equation (57), for q_zeta = 0, we have K_zeta = K_Y * sqrt(X**2 + Y**2)
        #
        # So then in equation (78):
        #   K_Y * sqrt(X^2 + Y^2) = - \Omega/c * R_ant * sin \varphi_t * cos \varphi_p
        #   => K_y = - \Omega/c * sin \varphi_t * cos \varphi_p
        #
        # since R_ant = sqrt(X^2 + Y^2)
        #
        K_Y_launch = (-wavenumber_K0
                         * np.sin( np.radians(toroidal_launch_angle_Torbeam)) 
                         * np.cos( np.radians(poloidal_launch_angle_Torbeam)) 
                         )
        
        # Use equation (58)
        K_Z_launch    = (-wavenumber_K0 
                         * np.sin( np.radians(poloidal_launch_angle_Torbeam))
                         )
        
        launch_K = np.array([K_X_launch, K_Y_launch, K_Z_launch])
        #----------------------------------------------------------------------
        
        
        
        #------------ TRANSFORMATION BETWEEN BEAM AND LAB FRAME ---------------
        #
        # 2x2 identity matrix
        identity_matrix_2D = np.eye(2, 2)
        
        # In TORBEAM, the toroidal angle is positive when measured anti-clockwise
        # from the NEGATIVE X-axis in the XY plane. Similarly, the poloidal angle is 
        # positive when measured anti-clockwise from the NEGATIVE R-axis in the RZ plane. 
        # For example, in the case of q_zeta = 0, the poloidal launch angle is measured
        # in the XZ plane, which means that the R-axis is the X-axis.
        #
        # Note that the beam is always launched from the outboard mid-plane inwards
        # (in the direction of negative R in the RZ plane) towards the origin.
        #
        # However, for this code, a toroidal angle is positive when measured 
        # anti-clockwise from the POSITIVE X-axis in the XY plane. Similarly, the 
        # poloidal angle is positive when measured anti-clockwise from the POSITIVE
        # R-axis in the RZ plane (e.g. X-axis in the XZ plane since q_zeta = 0). Therefore
        # our angle definition has a 180 degree phase difference as compared to the 
        # TORBEAM angles
        #
        # Then, using our angle definition, we want to launch a beam in the lab frame
        # which is of zero toroidal and zero poloidal angle (pointing towards the
        # positive X-axis) and then rotate the beam to where TORBEAM had specified
        # the launch angles (in our angle definition) via rotation matrices. These
        # are the poloidal and toroidal rotation angles that we have defined below
        toroidal_rotation_angle = np.radians(toroidal_launch_angle_Torbeam + 180)
        poloidal_rotation_angle = np.radians(poloidal_launch_angle_Torbeam + 180)
        
        # Toroidal rotation matrix to rotate about the Z-axis
        rotation_matrix_tor = np.array([
                                            [np.cos(toroidal_rotation_angle), np.sin(toroidal_rotation_angle), 0 ],
                                            [-np.sin(toroidal_rotation_angle), np.cos(toroidal_rotation_angle), 0 ],
                                            [0, 0, 1]
                                        ])
        
        # Poloidal rotation matrix to rotate about the Y-axis since our initial beam
        # is propagating in the positive X-axis direction
        rotation_matrix_pol = np.array([
                                            [ np.cos(poloidal_rotation_angle), 0, np.sin(poloidal_rotation_angle) ],
                                            [ 0, 1, 0 ],
                                            [ -np.sin(poloidal_rotation_angle), 0, np.cos(poloidal_rotation_angle) ]
                                            ])
        
        # We do toroidal rotation first followed by poloidal rotation. This is by
        # convention. In general, rotations are NOT commutative
        #
        # Note that rotation matrices are orthogonal matrices so the inverse is the transpose
        rotation_matrix         = np.matmul(rotation_matrix_pol,rotation_matrix_tor)
        rotation_matrix_inverse = np.transpose(rotation_matrix) 
        
        # Initialize Psi_w diagonal components given by equation (15) and (16) 
        # of Valerian's paper, noting that K = K_g at launch. This might seem
        # strange as in the paper, it is mentioned that the real and diagonal parts 
        # of Psi_w are not simultaneously diagonalizable. However, since we are 
        # working with circular beams, we can assume that it is.
        #
        # The way we have defined Psi_w_beam_launch is much different from the original
        # Scotty. In the original Scotty, we start of with a beam launched in the positive 
        # Z-direction and then, a rotation matrix is used to rotate a beam propagating in 
        # the positive Z-direction to the direction of propagation specified by the 
        # poloidal launch angle in TORBEAM. This is why in the original Scotty, 
        # the last row and last column of Psi_w_beam_launch is all zero.
        #
        # The approach above is not very good because the rotation of the K vector is from
        # the positive X-axis (need to check with Valerian why). Therefore, we have instead
        # chosen to set the first row and first column to be all zero in order to have a beam
        # propagating in the positive X-direction. This will make things simpler later on.
        #
        # Lastly, in the original Scotty, we divide the wavenumber_K0 by the radius of 
        # curvature of the launch beam. We do not do this here to avoid a division
        # by zero. For example, launching from a wave guide has a 0 radius of curvature
        Psi_w_beam_launch = np.zeros((2, 2), dtype = 'complex128')
        Psi_w_beam_launch[0][0] = wavenumber_K0 * launch_beam_curvature + 2j*launch_beam_width**(-2)
        Psi_w_beam_launch[1][1] = wavenumber_K0 * launch_beam_curvature + 2j*launch_beam_width**(-2)
        
        # Initialize Psi_3D from Psi_w.
        Psi_3D_beam_launch = np.zeros((3, 3), dtype = 'complex128')
        Psi_3D_beam_launch[1][1] = Psi_w_beam_launch[0][0]
        Psi_3D_beam_launch[2][2] = Psi_w_beam_launch[1][1]
       
        
        # Rotate our lab frame beam, propagating in the positive X-axis to the direction 
        # specified by TORBEAM's poloidal and toroidal launch angles (in our angle definition)
        Psi_3D_lab_launch = np.matmul(rotation_matrix_inverse, 
                                        np.matmul(Psi_3D_beam_launch, 
                                                  rotation_matrix) 
                                        )
        
        #----------------------------------------------------------------------
        
        
        #---------------------------------------------------------------------
        # CASE 1.1: Beam travels awhile through the vacuum before reaching the 
        #           plasma  
        #---------------------------------------------------------------------
        if vacuum_propagation_flag:
            
            print('Vacuum propagation turned ON')
            
            # locate entry point using find_entry_point
            R_step_magnitude = 0.1
            entry_position = find_entry_point(poloidal_flux_enter, 
                                              launch_position,
                                              poloidal_launch_angle_Torbeam,
                                              toroidal_launch_angle_Torbeam,
                                              find_poloidal_flux,
                                              R_step_magnitude)
            
            print(entry_position)
            
            
            # find the magnitude of the distance from the launch position
            # to the entry point
            distance_from_launch_to_entry = np.sqrt(
                                                    (launch_position[0] - entry_position[0])**2
                                                    + (launch_position[1] - entry_position[1])**2 
                                                    + (launch_position[2] - entry_position[2])**2
                                                    )
            
            
            ### CALCULATE ENTRY PARAMETERS FROM LAUNCH PARAMETERS ###
            #
            # The wavevector does not change in vacuum from the launch position
            # to the entry position
            K_X_entry = launch_K[0]
            K_Y_entry = launch_K[1]
            K_Z_entry = launch_K[2]
            Psi_w_beam_inverse_launch = find_inverse_2D(Psi_w_beam_launch)
            
            # This is the vacuum solution!
            Psi_w_beam_inverse_entry = (
                                         (distance_from_launch_to_entry/(wavenumber_K0))
                                        *identity_matrix_2D 
                                        + Psi_w_beam_inverse_launch
                                        )
            
            Psi_w_beam_entry = find_inverse_2D(Psi_w_beam_inverse_entry)
            
            # The formula for Psi_3D is given by equation (A.12) in Valerian's paper.
            # In this formula, only Psi_w contributes to the x(tau) and y(tau)
            # components while the components of Psi_3D along g attributes to 3 terms:
            #
            #   (a) g_hat/g (dK/dtau)
            #   (b) (dK_w/dtau)_w g_hat/g
            #   (c) K_g \kappa g_hat
            #
            # For terms (a) and (b), equation (A.30) can be used. At the entry
            # point, we are still travelling in vacuum and hence \grad H = 0
            # where H is the index of refraction squared (see Felix Parra notes),
            # therefore dK/dtau = 0 and correspondingly, (a) and (b) are zero.
            #
            # For term (c), we use the fact that \kappa = g^{-1} dg_hat/dtau
            # from equation (A.8). Since we always have |g_hat| = 1, dg_hat/dtau
            # can only change in terms of the direction.  However, in vacuum, 
            # the direction of g_hat is fixed (straight line)
            #
            # So the only contribution is Psi_w
            Psi_3D_beam_entry = np.array([
                    [0, 0, 0],
                    [0, Psi_w_beam_entry[0][0], Psi_w_beam_entry[0][1]],
                    [0, Psi_w_beam_entry[1][0], Psi_w_beam_entry[1][1]]
                    ])

            
        
            Psi_3D_lab_entry = np.matmul(rotation_matrix_inverse, 
                                                    np.matmul(Psi_3D_beam_entry, 
                                                              rotation_matrix) 
                                                    )
            
            
            ### FIND THE INITIAL PARAMETERS IN THE PLASMA ###
            #
            # Use continuity of the wavevector over the vacuum-plasma boundary to find
            # the initial wave vector just inside the plasma
            K_X_initial = K_X_entry 
            K_Y_initial = K_Y_entry
            K_Z_initial = K_Z_entry
            initial_position = entry_position
            
            #------------------------------------------------------------------
            # CASE 1.1.1: Boundary conditions used at vacuum-plasma boundary
            #------------------------------------------------------------------
            if Psi_BC_flag: 
                print('skip for now')
            
            #------------------------------------------------------------------
            # CASE 1.1.2: No boundary conditions used
            #------------------------------------------------------------------
            else: 
                Psi_3D_lab_initial = Psi_3D_lab_entry
                
        #---------------------------------------------------------------------
        # CASE 1.2: No vacuum propagation, beam is launched from vacuum and 
        #           immediately hits the plasma
        #---------------------------------------------------------------------
        else:
            print('Vacuum propagation turned OFF')  
            
            Psi_3D_lab_initial = Psi_3D_lab_launch
            K_X_initial = launch_K[0]
            K_Y_initial = launch_K[1]
            K_Z_initial = launch_K[2]
            initial_position = launch_position
            
            # delete all possible values of the entry point
            Psi_3D_lab_entry = None
            distance_from_launch_to_entry = None
            Psi_3D_lab_entry = np.full_like(Psi_3D_lab_launch,
                                            fill_value=np.nan)
    
    #-------------------------------------------------------------------------
    # CASE 2: Probe beam launched inside plasma
    #-------------------------------------------------------------------------
    
    else:
        
        print('Beam launched from inside the plasma')
        
        # use input parameters
        K_X_launch = plasmaLaunch_K[0]
        K_Y_launch = plasmaLaunch_K[1]
        K_Z_launch = plasmaLaunch_K[2]
        
        Psi_3D_lab_initial = plasmaLaunch_Psi_3D_lab
        K_X_initial = K_X_launch
        K_Y_initial = K_Y_launch
        K_Z_initial = K_Z_launch
        initial_position = launch_position

            
    
    #==========================================================================
    #                                 SOLVER 
    #==========================================================================
    
    #------------------------ INITIAL CONDITIONS FOR SOLVER -------------------
    
    beam_parameters_initial = np.zeros(18)
            
    beam_parameters_initial[0]  = initial_position[0] # q_X
    beam_parameters_initial[1]  = initial_position[1] # q_Y
    beam_parameters_initial[2]  = initial_position[2] # q_Z
     
    beam_parameters_initial[3]  = K_X_initial # K_X
    beam_parameters_initial[4]  = K_Y_initial # K_Y
    beam_parameters_initial[5]  = K_Z_initial # K_Z
    
    beam_parameters_initial[6]  = np.real(Psi_3D_lab_initial[0,0]) # Real(Psi_XX)
    beam_parameters_initial[7]  = np.real(Psi_3D_lab_initial[0,1]) # Real(Psi_XY)
    beam_parameters_initial[8]  = np.real(Psi_3D_lab_initial[0,2]) # Real(Psi_XZ)
    beam_parameters_initial[9]  = np.real(Psi_3D_lab_initial[1,1]) # Real(Psi_YY)
    beam_parameters_initial[10] = np.real(Psi_3D_lab_initial[1,2]) # Real(Psi_YZ)
    beam_parameters_initial[11] = np.real(Psi_3D_lab_initial[2,2]) # Real(Psi_ZZ)
    
    beam_parameters_initial[12] = np.imag(Psi_3D_lab_initial[0,0]) # Imag(Psi_XX)
    beam_parameters_initial[13] = np.imag(Psi_3D_lab_initial[0,1]) # Imag(Psi_XY) 
    beam_parameters_initial[14] = np.imag(Psi_3D_lab_initial[0,2]) # Imag(Psi_XZ)
    beam_parameters_initial[15] = np.imag(Psi_3D_lab_initial[1,1]) # Imag(Psi_YY)
    beam_parameters_initial[16] = np.imag(Psi_3D_lab_initial[1,2]) # Imag(Psi_YZ)
    beam_parameters_initial[17] = np.imag(Psi_3D_lab_initial[2,2]) # Imag(Psi_ZZ)  
    
    # If the ray hasn't left the plasma by the time this tau is reached, the 
    # solver gives up
    # tau_max = 10**5 (NOT NEEDED FOR BEAM)
    
    # Stuff the IVP solver needs to evolve beam_parameters, we do not include 
    # 'tau' and 'beam_parameters' as in beam_evolution_fun because these are the
    # independent and dependent variables respectively that will be keyed into
    # the solver.
    solver_arguments = (launch_angular_frequency, mode_flag,
                        delta_X, delta_Y, delta_Z, 
                        delta_K_X, delta_K_Y, delta_K_Z,
                        find_poloidal_flux, find_density_1D, 
                        find_B_X, find_B_Y, find_B_Z)
    
    # Set the range of tau needed for the output
    tau_leave = 10
    tau_points = np.linspace(0,tau_leave,102).tolist() 
    
    #------------------------ EXECUTE RK 4(5) SOLVER -------------------------
    
    print('Solver begins')
    
    # Start time where we start the solving (to time how long it takes)
    solver_start_time = time.time()
    
    # Invoke RK method of order 5(4) to solve the system of equations given by
    # (24), (25) and (27) in Valerian's paper.
    #
    # ARGUMENTS:
    # - beam_evolution_fun: RHS of the system of equations of (24), (25) and (27)
    # - [0, tau_leave]: range of the independent variable (which is tau)
    # - beam_parameters_initial: initial state of the (q_vec, K_vec, Psi) variables
    #                            on the LHS of the system of equations of (24), (25) and (27)
    # - method = 'RK45': tells the solver to use RK of order 5(4)
    # - t_eval = tau_points: where to store the computed solutions
    # - dense_output = False: tells solver NOT to compute continuous solution
    # - events = None: tells solver NOT to track any events
    # - vectorized = False: tells solver beam_eveolution_fun is NOT a vectorized 
    #                       function (see np.vectorize)
    # - args = solver_arguments: all other arguments in beam_evolution_fun that 
    #                            is not 'tau' and 'beam_parameters'
    #
    solver_beam_output = integrate.solve_ivp(
                                        beam_evolution_fun, 
                                        [0, tau_leave], 
                                        beam_parameters_initial, 
                                        method='BDF',
                                        t_eval = tau_points, 
                                        dense_output = False, 
                                        events = None,
                                        vectorized = False, 
                                        args = solver_arguments
                                        #rtol = 1e-1,
                                        #atol = 1e-4
                                    )
    
    # End time where the solver completes ()
    solver_end_time = time.time()
    
    # tells us how long the solver took
    print('Time taken (beam solver)', solver_end_time-solver_start_time,'s')
    
    #---------------------------- PROCESS SOLUTION ----------------------------
    
    # saves the solution 
    beam_parameters = solver_beam_output.y
    tau_array = solver_beam_output.t
    
    
    # if solver_status = 0, there was no issue solving the IVP. However,
    # if solver_status = +- 1, then there is an issue in the solver. 
    # See the documentation for more details
    solver_status = solver_beam_output.status
    
    numberOfDataPoints = len(tau_array)
    
    q_X_array = np.real(beam_parameters[0,:])
    q_Y_array = np.real(beam_parameters[1,:])
    q_Z_array = np.real(beam_parameters[2,:])
    
    K_X_array = np.real(beam_parameters[3,:])
    K_Y_array = np.real(beam_parameters[4,:])
    K_Z_array = np.real(beam_parameters[5,:])
    
    Psi_XX_real_array = beam_parameters[6,:]
    Psi_XY_real_array = beam_parameters[7,:]
    Psi_XZ_real_array = beam_parameters[8,:]
    Psi_YY_real_array = beam_parameters[9,:]
    Psi_YZ_real_array = beam_parameters[10,:]
    Psi_ZZ_real_array = beam_parameters[11,:]
    Psi_XX_imag_array = beam_parameters[12,:]
    Psi_XY_imag_array = beam_parameters[13,:]
    Psi_XZ_imag_array = beam_parameters[14,:]
    Psi_YY_imag_array = beam_parameters[15,:]
    Psi_YZ_imag_array = beam_parameters[16,:]
    Psi_ZZ_imag_array = beam_parameters[17,:]
    
    Psi_3D_output = np.zeros([numberOfDataPoints,3,3],dtype='complex128')
    Psi_3D_output[:,0,0] = Psi_XX_real_array + 1j * Psi_XX_imag_array # d (Psi_XX) / d tau
    Psi_3D_output[:,1,1] = Psi_YY_real_array + 1j * Psi_YY_imag_array # d (Psi_YY) / d tau
    Psi_3D_output[:,2,2] = Psi_ZZ_real_array + 1j * Psi_ZZ_imag_array # d (Psi_ZZ) / d tau
    Psi_3D_output[:,0,1] = Psi_XY_real_array + 1j * Psi_XY_imag_array # d (Psi_XY) / d tau
    Psi_3D_output[:,0,2] = Psi_XZ_real_array + 1j * Psi_XZ_imag_array # d (Psi_XZ) / d tau
    Psi_3D_output[:,1,2] = Psi_YZ_real_array + 1j * Psi_YZ_imag_array # d (Psi_YZ) / d tau
    Psi_3D_output[:,1,0] = Psi_3D_output[:,0,1]
    Psi_3D_output[:,2,0] = Psi_3D_output[:,0,2]
    Psi_3D_output[:,2,1] = Psi_3D_output[:,1,2]

    print('Main loop complete')
            
    #==========================================================================
    #                        SAVING THE SOLUTION
    #==========================================================================
    
    print('Saving data')    
    
    np.savez('data_input' + output_filename_suffix, 
              poloidalFlux_grid = poloidalFlux_grid,
              data_X_coord = data_X_coord, 
              data_Y_coord = data_Y_coord,
              data_Z_coord = data_Z_coord,
              poloidal_launch_angle_Torbeam = poloidal_launch_angle_Torbeam,
              toroidal_launch_angle_Torbeam = toroidal_launch_angle_Torbeam,
              launch_freq_GHz = launch_freq_GHz,
              mode_flag = mode_flag,
              launch_beam_width = launch_beam_width,
              launch_beam_curvature = launch_beam_curvature,
              launch_position = launch_position,
              launch_K = launch_K,
              ne_data_density_array = ne_data_density_array,
              ne_data_radialcoord_array = ne_data_radialcoord_array,
              equil_time = equil_time
             )    
    
    np.savez('solver_output' + output_filename_suffix, 
             solver_status = solver_status,
             tau_array = tau_array,
             q_X_array = q_X_array,
             q_Y_array = q_Y_array,
             q_Z_array = q_Z_array,
             K_X_array = K_X_array,
             K_Y_array = K_Y_array,
             K_Z_array = K_Z_array,
             Psi_3D_output = Psi_3D_output
             )   
    
    if solver_status == -1:
        # If the solver doesn't finish, end the function here
        print('Solver did not reach completion')
        return None
        


###############################################################################
#                                                                             #
#----------------------------- END OF CHECKPOINT 1 ---------------------------#
#                                                                             #
###############################################################################














