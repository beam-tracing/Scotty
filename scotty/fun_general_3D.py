import logging
from math import floor, log10
import numpy as np
from scotty.fun_general import find_D
from scotty.geometry_3D import MagneticField_3D_Cartesian
from scotty.typing import FloatArray
from typing import Union

log = logging.getLogger(__name__)

##################################################
#                                                #
# MISCELLANEOUS CODES                            #
#                                                #
##################################################

def find_beam_widths_curvs(
    Psi_w, K_vec, g_hat):
    
    Re_Psi_w = np.real(Psi_w)
    Re_Psi_w_eigvals = np.linalg.eigvalsh(Re_Psi_w)
    K_mag = np.linalg.norm(K_vec, axis=1)
    K_g_mag = np.sum(K_vec * g_hat, axis=1)
    curvs = (K_g_mag**2 / K_mag**3)[:, np.newaxis] * Re_Psi_w_eigvals

    Im_Psi_w = np.imag(Psi_w)
    Im_Psi_w_eigvals = np.linalg.eigvalsh(Im_Psi_w)
    widths = np.sqrt( np.full(Im_Psi_w_eigvals.shape, 2) / Im_Psi_w_eigvals )

    return curvs[:, 0], curvs[:, 1], widths[:, 0], widths[:, 1]



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
    
    if isinstance(K_magnitude, float): _flag = True
    elif len(K_magnitude) == 1:        _flag = True
    else:                              _flag = False

    if _flag:
        D_tensor = np.zeros([3, 3], dtype="complex128")
        D_tensor[0, 0] = D_11
        D_tensor[1, 1] = D_22
        D_tensor[2, 2] = D_bb
        D_tensor[0, 1] = -1j * D_12
        D_tensor[1, 0] = 1j * D_12
        D_tensor[0, 2] = D_1b
        D_tensor[2, 0] = D_1b
    else:
        D_tensor = np.zeros([len(K_magnitude), 3, 3], dtype="complex128")
        D_tensor[:, 0, 0] = D_11
        D_tensor[:, 1, 1] = D_22
        D_tensor[:, 2, 2] = D_bb
        D_tensor[:, 0, 1] = -1j * D_12
        D_tensor[:, 1, 0] = 1j * D_12
        D_tensor[:, 0, 2] = D_1b
        D_tensor[:, 2, 0] = D_1b

    return np.linalg.eigh(D_tensor)



def poloidal_flux_along_ray_line(X: float, Y: float, Z: float, tau: Union[float, int, FloatArray],
                                 poloidal_launch_angle_deg_Torbeam: float,
                                 toroidal_launch_angle_deg_Torbeam: float,
                                 field):
    
    positions = ray_line(X, Y, Z, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam)

    # Get the poloidal flux values at a particular point or point(s).
    # If there are NaNs, then replace those with the first non-NaN instance
    # in the array (i.e. the poloidal flux of the first point in the field)
    if positions.ndim == 1:
        polflux = field.polflux(*positions)
        if np.isnan(polflux):
            log.warning(f"Poloidal flux is NaN at [X,Y,Z] = [{X, Y, Z}]. Returning None instead")
            polflux = None
    if positions.ndim == 2:
        polflux = field.polflux(positions[:, 0], positions[:, 1], positions[:, 2])
        if np.all(np.isnan(polflux)):
            _printmsg = "\n".join(f"            - [{position}]" for position in positions)
            log.warning(f"""
        Poloidal fluxes are NaNs at all points queried: [X,Y,Z] = \n{_printmsg}

        Returning None instead. This may or may not cause issues
        """)
            polflux = None
        else:
            NaN_replacement_value = polflux[~np.isnan(polflux)][0]
            polflux = np.nan_to_num(polflux, nan=NaN_replacement_value)
    
    return polflux



def poloidal_flux_difference_along_ray_line(X: float, Y: float, Z: float, tau: Union[float, int, FloatArray],
                                            poloidal_launch_angle_deg_Torbeam: float,
                                            toroidal_launch_angle_deg_Torbeam: float,
                                            field,
                                            poloidal_flux_enter: float):
    """Signed poloidal flux distance to plasma boundary"""
    
    polflux = poloidal_flux_along_ray_line(X, Y, Z, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam, field)

    # If NaN, then just return NaN (saved as None)
    if polflux is None: return None
    else: return polflux - poloidal_flux_enter



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



def round_to_1sf(x: float, return_dp: bool = False):
    if x == 0: res = dp = 0
    else:
        dp = -int(floor(log10(abs(x))))
        res = round(x, dp)
    if return_dp: return res, dp
    else:         return res



def get_arr_bounds(xmin: float, xmax: float, spacing: float):
    if xmin > xmax: raise ValueError(f"`xmin` must be less than `mxax`")
    lnum = int((abs(xmin) // spacing) * np.sign(xmin)) - int(xmin<0)
    rnum = int((abs(xmax) // spacing) * np.sign(xmax)) + int(xmax>0)
    num = rnum - lnum + 1
    lb, rb = np.array([lnum, rnum]) * spacing
    return lb, rb, num