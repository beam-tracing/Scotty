# -*- coding: utf-8 -*-
"""
Created: 23/08/2022
Last Updated: 08/09/2022
@author: Tan Zheng Yang

Function that locates the entry point in which the beam propagating in vacuum 
enters the plasma.

Wrote it separately for the Scotty_beam_me_up_CART as it is too long
"""

### LOCATE ENTRY POINT OF BEAM INTO PLASMA ###
#
# In the original code, we have q_zeta = 0, meaning that Y = 0 at the launch 
# position. However, this the cartesian version of Scotty allows for arbitrary Y.


import numpy as np 
from Scotty_fun_general_CART import find_nearest 

def find_entry_point(poloidal_flux_enter, 
                     launch_position,
                     poloidal_launch_angle_Torbeam,
                     toroidal_launch_angle_Torbeam,
                     find_poloidal_flux,
                     R_step_magnitude):
    
    """
    DESCRIPTION
    ==============================
    Calculates the position where the beam enters the plasma.
    Utilizes the fact that on the magnetic axis (major axis for
    tokamaks), the poloidal flux is zero since the area of the flux
    surface there is zero. (Assuming circular flux surfaces, the
    flux surfaces are poloidal cross-sections, centered around the
    magnetic axis with radius equal to the radial distance from the
    magnetic axis)
    
    The calculation is done by using a while loop, using the variable
    'current_poloidal_flux' to track the value of the poloidal flux.
    At the point r_N where current_poloidal_flux < poloidal_flux_enter
    (meaning that the point r_N is inside the plasma), we impose a
    fine 3D rectangular mesh between the points r_{N-1} and r_N where
    r_{N-1} is outside the plasma (current_poloidal_flux > poloidal_flux_enter),
    and use a combination of find_poloidal_flux  and find_nearest
    to find the closest value to poloidal_flux_enter.
    
    INPUT
    ==============================
    poloidal_flux_enter (float):
        Value of the poloidal flux at the LCFS.
        
    launch_position (1x3 numpy array):
        Indicates the X, Y and Z position of the LAUNCH position.
        Note that Y = 0 if we are launching from q_zeta = 0
        
    poloidal_launch_angle_Torbeam (float):
        The poloidal launch angle of the beam at the launch position. Calculated
        using TORBEAM.
        
    toroidal_launch_angle_Torbeam (float):
        The poloidal launch angle of the beam at the launch position. Calculated
        using TORBEAM.
        
    find_poloidal_flux (function):
        Finds the value of the poloidal flux given q_X, q_Y and q_Z
        
    R_step_magnitude (positive float):
        The magnitude of the spatial step in the R-direction. 
        
    OUTPUT
    ==============================
    entry_position (1x3 numpy array):
        Indicates the X, Y and Z position of the ENTRY position.
        Note that Y = 0 since we are launching from q_zeta = 0
    """
    
    launch_X = launch_position[0]
    launch_Y = launch_position[1]
    launch_Z = launch_position[2]
    
    # Point n in iteration (starts from launch position)
    prev_X = launch_X
    prev_Y = launch_Y
    prev_Z = launch_Z
    
    # Point n+1 in iteration
    current_X = 0
    current_Y = 0
    current_Z = 0
    
    # Our beam is launched from the outboard position, the beam is moving in the 
    # direction of negative R. Therefore, our R_step should be negative
    R_step = -R_step_magnitude 
    
    # Convert the TORBEAM angles (poloidal and toroidal measured anti-clockwise
    # from NEGATIVE X-axis) to the angles in our system (measured anti-clockwise
    # from POSITIVE X-axis)
    phi_t = np.radians(toroidal_launch_angle_Torbeam + 180)
    phi_p = np.radians(poloidal_launch_angle_Torbeam + 180)
    
    # The individual X_step, Y_step and Z_step can be found using equations (78)
    # of Valerians paper but for positions.
    #  
    # Technically, the SIGN of X_step, Y_step and Z_step depends on the launch
    # position. For example, if launch_X < 0, then X_step > 0 however if
    # launch_X > 0 then X_step < 0. However, this is taken care of by the 
    # toroidal and poloidal angles. For example, if launch_X < 0, then clearly
    # we have cos(phi_t) < 0 and therefore since R_step < 0, we have X_step > 0 
    X_step = R_step * np.cos(phi_t) * np.cos(phi_p)
    Y_step = R_step * np.sin(phi_t) * np.cos(phi_p)
    Z_step = R_step * np.sin(phi_p)
    
    # Calculates the initial flux. "grid = False" ensures we get back a 1D array
    # corresponding to each index of the current_X, current_Y and current_Z arrays
    #
    # NO "grid = False" argument, since our interpolator is RegularGridInterpolator
    # instead of RectBivariateSpline
    
    current_poloidal_flux = find_poloidal_flux(prev_X, prev_Y, prev_Z)
    
    # While loop to find a rough estimate of the first point in the plasma
    while current_poloidal_flux > poloidal_flux_enter:
        
        # New [X, Y, Z] after taking steps in the X, Y and Z direction corresponding
        # to a R_step in the negative R direction. Remember that the
        # signs of X_step, Y_step and Z_step are already handled by the toroidal
        # and poloidal angles
        current_X = prev_X + X_step
        current_Y = prev_Y + Y_step
        current_Z = prev_Z + Z_step
        
        # Computes the poloidal flux at the new point
        current_poloidal_flux = find_poloidal_flux(current_X, current_Y, current_Z)
        
        # CASE 1: Current point is already the entry point (DONE)
        if current_poloidal_flux == poloidal_flux_enter:
            entry_position = np.array([current_X, 0, current_Z])
            return entry_position
        
        # CASE 2: Current point is inside the plasma (break while loop)
        elif current_poloidal_flux < poloidal_flux_enter:
            break
        
        # CASE 3: Current point is still outside the plasma (continue while loop)
        else:
            prev_X = current_X 
            prev_Y = current_Y
            prev_Z = current_Z
            continue
        
    #--------------------------------------------------------------------------
    #                            IF CASE 2 OCCURS
    #--------------------------------------------------------------------------
            
    # impose fine mesh in between the points (X_prev, Y_prev, Z_prev)
    # and (current_X, current_Y, current_Z)
    num_fine_search_points = 1000
    fine_mesh_X = np.linspace(prev_X, current_X, num_fine_search_points)
    fine_mesh_Y = np.linspace(prev_Y, current_Y, num_fine_search_points)
    fine_mesh_Z = np.linspace(prev_Z, current_Z, num_fine_search_points)

    # find the value of the poloidal fluxes in this fine mesh
    fine_mesh_poloidal_flux = find_poloidal_flux(fine_mesh_X, fine_mesh_Y, fine_mesh_Z)
    
    # finds the index of the first value in fine_mesh_poloidal_flux that is
    # closest to poloidal_flux_enter
    entry_index = find_nearest(fine_mesh_poloidal_flux,
                               poloidal_flux_enter)
    
    if fine_mesh_poloidal_flux[entry_index] > poloidal_flux_enter:
        # The first point needs to be in the plasma. If the first point is
        # outside, then there will be errors when the gradients are calculated
        #
        # note that this requires the poloidal flux to be monotone increasing 
        # radially from the magnetic axis
        entry_index = entry_index + 1
        
    entry_position = np.array([
                                fine_mesh_X[entry_index], 
                                fine_mesh_Y[entry_index], 
                                fine_mesh_Z[entry_index]
                                ])
    
    print('entry found!')
    return entry_position
