import numpy as np
from scotty.fun_general import find_D
from scotty.geometry_3D import MagneticField_3D_Cartesian
from scotty.typing import FloatArray
from typing import Union
import time

##################################################
#                                                #
# MISCELLANEOUS CODES                            #
#                                                #
##################################################

def timer(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    return result, duration



def find_H_Cardano_eig(
    K_magnitude,
    launch_angular_frequency,
    epsilon_para,
    epsilon_perp,
    epsilon_g,
    theta_m):

    D_11, D_22, D_bb, D_12, D_1b = find_D(
        K_magnitude,
        launch_angular_frequency,
        epsilon_para,
        epsilon_perp,
        epsilon_g,
        theta_m)
    
    D_tensor = np.zeros([3, 3], dtype="complex128")
    D_tensor[0, 0] = D_11
    D_tensor[1, 1] = D_22
    D_tensor[2, 2] = D_bb
    D_tensor[0, 1] = -1j * D_12
    D_tensor[1, 0] = 1j * D_12
    D_tensor[0, 2] = D_1b
    D_tensor[2, 0] = D_1b

    return np.linalg.eigh(D_tensor)



def ray_line(X: float, Y: float, Z: float, tau: Union[float, int, FloatArray],
             poloidal_launch_angle_deg_Torbeam: float,
             toroidal_launch_angle_deg_Torbeam: float):
    
    XYZ_start = np.array([X, Y, Z])

    # This parametrises the ray in a line normal to the antenna,
    # at a/up to some values of the parameter `tau`.
    poloidal_launch_angle = -np.deg2rad(poloidal_launch_angle_deg_Torbeam)
    toroidal_launch_angle =  np.deg2rad(toroidal_launch_angle_deg_Torbeam) + np.pi
    X_step = np.cos(toroidal_launch_angle) * np.cos(poloidal_launch_angle)
    Y_step = np.sin(toroidal_launch_angle) * np.cos(poloidal_launch_angle)
    Z_step = np.sin(poloidal_launch_angle)
    XYZ_step = np.array((X_step, Y_step, Z_step))

    ray_line_positions = XYZ_start + np.outer(tau, XYZ_step)

    if np.ndim(tau) == 0: return ray_line_positions[0]
    else:                 return ray_line_positions



def poloidal_flux_along_ray_line(X: float, Y: float, Z: float, tau: Union[float, int, FloatArray],
                                 poloidal_launch_angle_deg_Torbeam: float,
                                 toroidal_launch_angle_deg_Torbeam: float,
                                 field: MagneticField_3D_Cartesian):
    
    position = ray_line(X, Y, Z, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam)

    # Get the poloidal flux values at a particular point or point(s).
    # If there are NaNs, then replace those with the first non-NaN instance
    # in the array (i.e. the poloidal flux of the first point in the field)
    if position.ndim == 1:
        polflux = field.polflux(*position)
    if position.ndim == 2:
        polflux = np.field.polflux(position[:, 0], position[:, 1], position[:, 2])
        NaN_replacement_value = polflux[~np.isnan(polflux)][0]
        polflux = np.nan_to_num(polflux, nan=NaN_replacement_value)
    
    return polflux



def poloidal_flux_difference_along_ray_line(X: float, Y: float, Z: float, tau: Union[float, int, FloatArray],
                                            poloidal_launch_angle_deg_Torbeam: float,
                                            toroidal_launch_angle_deg_Torbeam: float,
                                            field: MagneticField_3D_Cartesian,
                                            poloidal_flux_enter):
    """Signed poloidal flux distance to plasma boundary"""
    
    polflux = poloidal_flux_along_ray_line(X, Y, Z, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam, field)
    
    return polflux - poloidal_flux_enter