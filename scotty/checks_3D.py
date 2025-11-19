# -*- coding: utf-8 -*-
# Copyright 2022 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""Checks input arguments of `beam_me_up <scotty.beam_me_up.beam_me_up>`"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from scotty.fun_general import angular_frequency_to_wavenumber, freq_GHz_to_angular_frequency
from scotty.geometry_3D import MagneticField_3D_Cartesian
from scotty.profile_fit import ProfileFitLike
from scotty.typing import FloatArray
import numpy as np
from typing import Literal, Optional, Sequence, Union

log = logging.getLogger(__name__)

##################################################
#
# CLASS PARAMETERS
#
##################################################

class Parameters:
    r"""
    A place to store all important information

    1. Should add a "flag" subclass?
    2. Should create a new Python file for input checks
    """

    def __init__(
        self,

        poloidal_launch_angle_Torbeam: float,
        toroidal_launch_angle_Torbeam: float,
        mode_flag_launch: Literal["O", "X", -1, 1],
        launch_freq_GHz: float,
        launch_beam_width: float,
        launch_beam_curvature: float,

        q_launch_cartesian: FloatArray,

        vacuumLaunch_flag: bool,
        vacuum_propagation_flag: bool,
        Psi_BC_flag: Optional[str],

        poloidal_flux_enter: float,
        poloidal_flux_zero_density: float,
        poloidal_flux_zero_temperature: float,

        auto_delta_sign: bool,
        delta_X: float,
        delta_Y: float,
        delta_Z: float,
        delta_K_X: float,
        delta_K_Y: float,
        delta_K_Z: float,
        len_tau: int,
        rtol: float,
        atol: float,

        interp_order: Union[str, int],
        interp_smoothing: int,
        density_fit_parameters: Optional[Sequence],
        density_fit_method: Optional[Union[str, ProfileFitLike]],
        temperature_fit_parameters: Optional[Sequence],
        temperature_fit_method: Optional[Union[str, ProfileFitLike]],

        magnetic_data_path: Union[str, Path],
        ne_data_path: Union[str, Path],
        Te_data_path: Union[str, Path],
        input_filename_suffix: str,
        output_path: Union[str, Path],
        output_filename_suffix: str,

        # verbose_run: bool, # TO REMOVE?
        # log_level: Union[str, int], # TO REMOVE?

        K_plasmaLaunch_cartesian: FloatArray,
        Psi_3D_plasmaLaunch_labframe_cartesian: FloatArray,
        ):

        log.debug(f"Creating `Parameters` class object")

        # TORBEAM antenna angles are anti-clockwise from negative X-axis,
        # so we need to rotate the toroidal angle by pi. This will take
        # care of the direction of the beam. The poloidal angle is also
        # reversed from its usual sense, so we can just flip it by adding
        # a minus sign
        self.poloidal_launch_angle_deg_Torbeam = poloidal_launch_angle_Torbeam
        self.poloidal_launch_angle_rad_Torbeam = np.deg2rad(self.poloidal_launch_angle_deg_Torbeam)
        self.toroidal_launch_angle_deg_Torbeam = toroidal_launch_angle_Torbeam
        self.toroidal_launch_angle_rad_Torbeam = np.deg2rad(self.toroidal_launch_angle_deg_Torbeam)
        # self.poloidal_launch_angle_rad = -self.poloidal_launch_angle_rad_Torbeam
        # self.poloidal_launch_angle_deg = np.rad2deg(self.poloidal_launch_angle_rad)
        # self.toroidal_launch_angle_rad = self.toroidal_launch_angle_rad_Torbeam + np.pi
        # self.toroidal_launch_angle_deg = np.rad2deg(self.toroidal_launch_angle_rad)

        self.mode_flag_launch = mode_flag_launch
        self.mode_flag_initial = None
        self.launch_frequency_GHz = launch_freq_GHz
        self.launch_angular_frequency = freq_GHz_to_angular_frequency(self.launch_frequency_GHz)
        self.launch_wavenumber = angular_frequency_to_wavenumber(self.launch_angular_frequency)
        self.launch_beam_width = launch_beam_width
        self.launch_beam_curvature = launch_beam_curvature

        self.q_launch_cartesian = q_launch_cartesian
        self.q_initial_cartesian = None # Only for hinting

        # TO REMOVE -- can introduce this after removing the vacuumLaunch processing from the main beam_me_up_3D routine
        # self.K_plasmaLaunch_cartesian = K_plasmaLaunch_cartesian
        # self.Psi_3D_plasmaLaunch_labframe_cartesian = Psi_3D_plasmaLaunch_labframe_cartesian
        
        self.vacuumLaunch_flag = vacuumLaunch_flag
        self.vacuum_propagation_flag = vacuum_propagation_flag
        self.Psi_BC_flag = Psi_BC_flag
        
        self.poloidal_flux_enter = poloidal_flux_enter
        self.poloidal_flux_zero_density = poloidal_flux_zero_density
        self.poloidal_flux_zero_temperature = poloidal_flux_zero_temperature

        self.auto_delta_sign = auto_delta_sign
        self.delta_X = delta_X
        self.delta_Y = delta_Y
        self.delta_Z = delta_Z
        self.delta_K_X = delta_K_X
        self.delta_K_Y = delta_K_Y
        self.delta_K_Z = delta_K_Z
        self.len_tau = len_tau
        self.rtol = rtol
        self.atol = atol

        self.interp_order_magnetic_data = interp_order
        self.interp_order_ne_data = interp_order
        self.interp_smoothing = interp_smoothing
        self.density_fit_parameters = density_fit_parameters
        self.density_fit_method = density_fit_method
        self.temperature_fit_parameters = temperature_fit_parameters
        self.temperature_fit_method = temperature_fit_method

        self.magnetic_data_path = magnetic_data_path
        self.ne_data_path = ne_data_path
        self.ne_filename = None # only for hinting
        self.Te_data_path = Te_data_path
        self.input_filename_suffix = input_filename_suffix
        self.output_path = output_path
        self.output_filename_suffix = output_filename_suffix

        # self.verbose_run = verbose_run # TO REMOVE?

        self.K_plasmaLaunch_cartesian = K_plasmaLaunch_cartesian
        self.Psi_3D_plasmaLaunch_labframe_cartesian = Psi_3D_plasmaLaunch_labframe_cartesian
    
    def set_experimental_profiles(self):
        log.info(f"Setting experimental profiles")
        if (self.density_fit_parameters is None) and (self.density_fit_method in [None, "smoothing-spline-file"]):
            self.ne_filename = self.ne_data_path / f"ne{self.input_filename_suffix}.dat"
            self.density_fit_parameters = [self.ne_filename, self.interp_order_ne_data, self.interp_smoothing]

            # FIXME: Read data so it can be saved later
            ne_data = np.fromfile(self.ne_filename, dtype=float, sep="   ")
            # ne_data_density_array = ne_data[2::2]
            # ne_data_radialcoord_array = ne_data[1::2]
        else: self.ne_filename = None



##################################################
#
# CHECKING INPUTS (for beam_me_up_3D)
#
# TODO:
#    1) Need to put data pathway stuff here
#
#    2a) Put the density_fit_parameters and
#        density_fit_method stuff here
#
#    2b) Put ne_data checks here:
#        - rho must be sorted
#        - ne must be sorted???
#        - must polflux_zero_density be less than
#          the last polflux coord in ne.dat?
#
#    3) Need to put temperature data stuff here
#
##################################################

def check_user_inputs(params: Parameters):
    f"""Checking that all user inputs are valid."""

    # launch frequency > 0
    if params.launch_frequency_GHz <= 0:
        raise ValueError(f"`launch_freq_GHz` must be positive, but got `{params.launch_frequency_GHz}` GHz")
    
    # launch beam width > 0
    if params.launch_beam_width <= 0:
        raise ValueError(f"`launch_beam_width` must be positive, but got `{params.launch_beam_width}` m")
    
    # mode flag must be one of {_valid_launch_mode_flags}
    _valid_launch_mode_flags = [1, -1, "O", "X"]
    log.debug(f"`_valid_launch_mode_flags` set to {_valid_launch_mode_flags}")
    if params.mode_flag_launch not in _valid_launch_mode_flags:
        raise ValueError(f"`mode_flag` must be one of {_valid_launch_mode_flags}, but got `{params.mode_flag_launch}`")
    
    # Psi BC flag must be one of {_valid_Psi_BC_flags}
    _valid_Psi_BC_flags = ["discontinuous", "continuous", None]
    log.debug(f"`_valid_Psi_BC_flags` set to {_valid_Psi_BC_flags}")
    if params.Psi_BC_flag in [True, False]:
        raise ValueError(f"`Psi_BC_flag` = True or False is deprecated")
    elif params.Psi_BC_flag not in _valid_Psi_BC_flags:
        raise ValueError(f"`Psi_BC_flag` must be one of {_valid_Psi_BC_flags}, but got `{params.Psi_BC_flag}`")
    
    # poloidal flux enter must be positive
    if params.poloidal_flux_enter <= 0:
        raise ValueError(f"`poloidal_flux_enter` must be positive, but got `{params.poloidal_flux_enter}`")
    
    # poloidal flux zero density must be positive
    if params.poloidal_flux_zero_density <= 0:
        raise ValueError(f"`poloidal_flux_zero_density` must be positive, but got `{params.poloidal_flux_zero_density}`")

    # poloidal flux zero density must be positive
    if params.poloidal_flux_zero_temperature <= 0:
        raise ValueError(f"`poloidal_flux_zero_temperature` must be positive, but got `{params.poloidal_flux_zero_temperature}`")
    
    # Check that the poloidal flux arguments are valid. Specifically:
    # The `poloidal_flux_zero_density` (the poloidal flux value beyond which
    # the electron density is set to zero) must be greater than or equal to the
    # `poloidal_flux_enter` (the poloidal flux value where the ray enters the
    # plasma).
    if params.poloidal_flux_zero_density < params.poloidal_flux_enter:
        raise ValueError(f"`poloidal_flux_zero_density` is less than `poloidal_flux_enter`")

    # interp order for magnetic data must be one of
    # {_valid_interp_orders_int} or one of {_valid_interp_orders_str}
    _valid_interp_orders_int = [1, 3, 5]
    _valid_interp_orders_str = ["linear", "cubic", "quintic"]
    _valid_interp_orders = _valid_interp_orders_int + _valid_interp_orders_str
    log.debug(f"`_valid_interp_orders` set to {_valid_interp_orders}")
    if params.interp_order_magnetic_data in _valid_interp_orders_int:
        if   params.interp_order_magnetic_data == 1: params.interp_order_magnetic_data = "linear"
        elif params.interp_order_magnetic_data == 3: params.interp_order_magnetic_data = "cubic"
        elif params.interp_order_magnetic_data == 5: params.interp_order_magnetic_data = "quintic"
    elif params.interp_order_magnetic_data in _valid_interp_orders_str: pass
    else:
        log.warning(f"`interp_order` must be one of {_valid_interp_orders}, but got {params.interp_order_magnetic_data}")
        log.warning(f"Setting `interp_order` = `quintic` for interpolation")
        params.interp_order_magnetic_data = "quintic"
    
    # interp order for ne data must be one of
    # {_valid_interp_orders_int} or one of {_valid_interp_orders_str}
    if params.interp_order_ne_data in _valid_interp_orders_int: pass
    elif params.interp_order_ne_data in _valid_interp_orders_str:
        if   params.interp_order_ne_data == "linear":  params.interp_order_ne_data = 1
        elif params.interp_order_ne_data == "cubic":   params.interp_order_ne_data = 3
        elif params.interp_order_ne_data == "quintic": params.interp_order_ne_data = 5
    else:
        log.warning(f"`interp_order` must be one of {_valid_interp_orders}, but got {params.interp_order_ne_data}")
        log.warning(f"Setting `interp_order` = `quintic` for interpolation")
        params.interp_order_ne_data = 5
    
    # len tau must be positive
    if params.len_tau <= 0:
        log.warning(f"`len_tau` must be a positive integer, but got `{params.len_tau}`")
        log.warning(f"Setting `len_tau` = 102")
        params.len_tau = 102
    
    # file paths must be valid and exist
    # only checks for magnetic, ne, and Te data
    params.magnetic_data_path = Path(params.magnetic_data_path)
    params.ne_data_path = Path(params.ne_data_path)
    params.Te_data_path = Path(params.Te_data_path)
    params.output_path = Path(params.output_path)
    if not params.magnetic_data_path.is_dir(): raise ValueError(f"`magnetic_data_path` must be a valid directory")
    if not params.ne_data_path.is_dir(): raise ValueError(f"`ne_data_path` must be a valid directory")
    if not params.Te_data_path.is_dir(): raise ValueError(f"`Te_data_path` must be a valid directory")
    if not params.output_path.is_dir():
        log.warning(f"File path {params.output_path} does not exist. Creating the folder now")
        os.makedirs(params.output_path)
        # raise ValueError(f"`output_path` must be a valid directory")

    # file input and output suffixes must be strings
    params.input_filename_suffix = str(params.input_filename_suffix)
    params.output_filename_suffix = str(params.output_filename_suffix)



def _check_launch_position(poloidal_flux_enter: float, launch_position: FloatArray, field: MagneticField_3D_Cartesian) -> None:
    X, Y, Z = launch_position
    launch_psi = field.poloidal_flux(X, Y, Z)
    if launch_psi < poloidal_flux_enter:
        raise ValueError("Launch position (X={X:.4f}, Y={Y:.4f}, Z={Z:.4f}, psi={launch_psi:.4f}) is inside plasma (psi={poloidal_flux_enter})")



def _check_mode_flag_for_Psi_BC_flag(params: Parameters) -> None:
    """If Psi_BC_flag is None, then mode_flag must be only one of [1, -1].

    This condition is required because we avoid calculating some
    of the required quantities when we do not apply any BCs, so we
    simply pass the user's mode_flag as the mode_flag_initial (i.e.
    the mode flag to use when starting the Hamiltonian calculations)
    and this requires the mode flag to be explicitly either 1 or -1.

    TODO: User should be able to specify O- or X-mode as the mode_flag
    directly. Currently not implemented because the way the code is
    currently implemented does not allow one to directly check which
    solution corresponds to which mode.
    """
    log.debug(f"Checking mode_flag for Psi_BC_flag")

    _valid_initial_mode_flags = [1, -1]
    log.debug(f"`_valid_initial_mode_flags` set to {_valid_initial_mode_flags}")
    if params.mode_flag_initial not in _valid_initial_mode_flags and params.Psi_BC_flag is None:
        raise ValueError(f"`mode_flag` must be one of {_valid_initial_mode_flags} if `vacuumLaunch_flag` is False, but got `{params.mode_flag_initial}`")



def check_input_before_ray_tracing(params: Parameters):
    log.info(f"Checking the validity of user-specified arguments")
    _check_mode_flag_for_Psi_BC_flag(params)

    # Temporarily removing this function as the behaviour it checks for
    # is allowed in the new version which has poloidal_flux_enter and poloidal_zero_density
    # as separate input arguments
    # _check_launch_position(poloidal_flux_enter, launch_position, field)