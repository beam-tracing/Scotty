import logging
import numpy as np
from scotty.beam_me_up import beam_me_up
from scotty.beam_me_up_3D_temp import beam_me_up_3D
from scotty.fun_general_3D import round_to_1sf, get_arr_bounds
from scotty.typing import FloatArray
from typing import Optional, Tuple

log = logging.getLogger(__name__)

def benchmark_me_up_3D(beam_me_up_kwargs: dict, beam_me_up_3D_kwargs: dict):

    # Defining stuff here for ease
    _pad = 20

    # Pre-processing the dictionaries
    if "create_magnetic_geometry_3D" not in beam_me_up_3D_kwargs:
        beam_me_up_3D_kwargs["create_magnetic_geometry_3D"] = {}

    (cyl_dt,
     cyl_field,
     cyl_ne,
     cyl_Te,
     cyl_H) = beam_me_up(**beam_me_up_kwargs)
    
    # Converting the ray trajectory from cylindrical to cartesian
    # to obtain bounding box for 3d field
    cyl_q_R    = np.array(cyl_dt["solver_output"]["q_R"])
    cyl_q_zeta = np.array(cyl_dt["solver_output"]["q_zeta"])
    cyl_q_Z    = np.array(cyl_dt["solver_output"]["q_Z"])
    cyl_q_X = cyl_q_R * np.cos(cyl_q_zeta)
    cyl_q_Y = cyl_q_R * np.sin(cyl_q_zeta)

    # TO REMOVE -- 4 Dec, just for debugging
    # print("cyl_q_X", cyl_q_X)
    # print()
    # print("cyl_q_Y", cyl_q_Y)
    # print()
    # print("cyl_q_Z", cyl_q_Z)

    # Getting the keyword arguments for the 3d field
    R_coords = cyl_field.R_coord
    Z_coords = cyl_field.Z_coord
    cyl_R_spacing = abs(round_to_1sf(R_coords[1] - R_coords[0]))
    cyl_Z_spacing = abs(round_to_1sf(Z_coords[1] - Z_coords[0]))
    cart_X_spacing = beam_me_up_3D_kwargs["create_magnetic_geometry_3D"].pop("X_spacing", cyl_R_spacing)
    cart_Y_spacing = beam_me_up_3D_kwargs["create_magnetic_geometry_3D"].pop("Y_spacing", cyl_R_spacing)
    cart_Z_spacing = beam_me_up_3D_kwargs["create_magnetic_geometry_3D"].pop("Z_spacing", cyl_Z_spacing)
    pad = beam_me_up_3D_kwargs["create_magnetic_geometry_3D"].pop("pad", _pad)

    # Finding and constructing the bounding box ranges by padding the
    # ranges from before with 10 extra points on all sides because
    # interpolation requires at least 6 more points to be padded
    # and then saving them as entries of a kwarg dictionary. We also
    # check to make sure the padded points do not exit the bounds of
    # the original field
    def find_padded_array(arr: FloatArray, userpad: int, spacing: float, coord_arr: FloatArray = None) -> np.ndarray:
        _pad = max(userpad, pad)
        _min_coord = min(arr) - spacing*_pad
        _max_coord = max(arr) + spacing*_pad
        if coord_arr is not None: return np.array([x for x in coord_arr if _min_coord <= x <= _max_coord])
        else:                     return np.linspace(*get_arr_bounds(_min_coord, _max_coord, spacing))
    
    if "X" not in beam_me_up_3D_kwargs["create_magnetic_geometry_3D"]:
        _temp = find_padded_array(cyl_q_X, pad, cart_X_spacing, R_coords)
        beam_me_up_3D_kwargs["create_magnetic_geometry_3D"]["X"] = _temp
        log.debug(f"Creating X-coordinates: {_temp}")
    
    if "Y" not in beam_me_up_3D_kwargs["create_magnetic_geometry_3D"]:
        _temp = find_padded_array(cyl_q_Y, pad, cart_Y_spacing)
        beam_me_up_3D_kwargs["create_magnetic_geometry_3D"]["Y"] = _temp
        log.debug(f"Creating Y-coordinates: {_temp}")
    
    if "Z" not in beam_me_up_3D_kwargs["create_magnetic_geometry_3D"]:
        _temp = find_padded_array(cyl_q_Z, pad, cart_Z_spacing, Z_coords)
        beam_me_up_3D_kwargs["create_magnetic_geometry_3D"]["Z"] = _temp
        log.debug(f"Creating Z-coordinates: {_temp}")
    
    beam_me_up_3D_kwargs["find_B_method"] = cyl_field

    (cart_dt,
     cart_field,
     cart_ne,
     cart_Te,
     cart_H) = beam_me_up_3D(**beam_me_up_3D_kwargs)

    return (cyl_dt, cyl_field, cyl_ne, cyl_Te, cyl_H), (cart_dt, cart_field, cart_ne, cart_Te, cart_H)