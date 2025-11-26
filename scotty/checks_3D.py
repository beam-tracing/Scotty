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

VALID_LAUNCH_MODE_FLAGS = Literal[1, -1, "O", "X"]
VALID_LAUNCH_FLAGS = Literal[None, "outside", "boundary", "inside"] # TO REMOVE -- not implemtned yet, to be implemented when we combine vacuumLaunch and vacuum_prop
VALID_PSI_BC_FLAGS = Literal[None, "continuous", "discontinuous"]

class Parameters:
    r"""
    A place to store all important information

    1. Should add a "flag" subclass?
    2. Should create a new Python file for input checks
    """

    def __init__(
        self,

        # General launch parameters
        poloidal_launch_angle_Torbeam: float,
        toroidal_launch_angle_Torbeam: float,
        launch_freq_GHz: float,
        launch_beam_width: float,
        launch_beam_curvature: float,
        q_launch_cartesian: FloatArray,
        mode_flag_launch: VALID_LAUNCH_MODE_FLAGS,
        vacuumLaunch_flag: bool,
        vacuum_propagation_flag: bool,
        Psi_BC_flag: VALID_PSI_BC_FLAGS,
        relativistic_flag: bool,

        # Data input and output arguments
        magnetic_data_path: Union[str, Path],
        ne_data_path: Union[str, Path],
        Te_data_path: Union[str, Path],
        input_filename_suffix: str,
        output_path: Union[str, Path],
        output_filename_suffix: str,
        shot: Optional[int],
        equil_time: Optional[Union[int, float]],

        # Solver settings and finite-difference parameters
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
        poloidal_flux_enter: float,
        poloidal_flux_zero_density: float,
        poloidal_flux_zero_temperature: float,

        # Interpolation settings
        density_fit_parameters: Optional[Sequence],
        density_fit_method: Optional[Union[str, ProfileFitLike]],
        temperature_fit_parameters: Optional[Sequence],
        temperature_fit_method: Optional[Union[str, ProfileFitLike]],
        interp_order: Union[str, int],
        interp_smoothing: int,

        # Plotting flags
        figure_flag: bool,
        further_analysis_flag: bool,
        detailed_analysis_flag: bool,

        # Additional flags
        quick_run: bool,
        return_dt_field: bool,

        # Extra kwargs for parsing
        **kwargs,
        ):

        # log.debug(f"Creating `Parameters` class object")

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

        self.launch_frequency_GHz = self._check_positive("launch_freq_GHz", launch_freq_GHz)
        self.launch_angular_frequency = freq_GHz_to_angular_frequency(self.launch_frequency_GHz)
        self.launch_wavenumber = angular_frequency_to_wavenumber(self.launch_angular_frequency)
        self.launch_beam_width = self._check_positive("launch_beam_width", launch_beam_width)
        self.launch_beam_curvature = launch_beam_curvature
        self.q_launch_cartesian = q_launch_cartesian
        self.q_initial_cartesian = None # Initialising first
        self.mode_flag_launch = mode_flag_launch
        self.mode_flag_initial = None # Initialising first
        self.vacuumLaunch_flag = vacuumLaunch_flag
        self.vacuum_propagation_flag = vacuum_propagation_flag
        self.Psi_BC_flag = Psi_BC_flag
        self.relativistic_flag = relativistic_flag

        self.magnetic_data_path = self._check_path("magnetic", magnetic_data_path)
        self.ne_data_path = self._check_path("ne", ne_data_path)
        self.ne_filename = None # Initialising first
        self.Te_data_path = self._check_path("Te", Te_data_path)
        self.input_filename_suffix = input_filename_suffix
        self.output_path = self._check_path("output", output_path)
        self.output_filename_suffix = output_filename_suffix
        self.shot = shot
        self.equil_time = equil_time
        
        self.auto_delta_sign = auto_delta_sign
        self.delta_X = delta_X
        self.delta_Y = delta_Y
        self.delta_Z = delta_Z
        self.delta_K_X = delta_K_X
        self.delta_K_Y = delta_K_Y
        self.delta_K_Z = delta_K_Z
        self.len_tau = self._check_positive("len_tau", len_tau)
        self.rtol = rtol
        self.atol = atol
        self.poloidal_flux_enter = self._check_positive("poloidal_flux_enter", poloidal_flux_enter)
        self.poloidal_flux_zero_density = self._check_positive("poloidal_flux_zero_density", poloidal_flux_zero_density)
        self.poloidal_flux_zero_temperature = self._check_positive("poloidal_flux_zero_temperature", poloidal_flux_zero_temperature)

        self.density_fit_parameters = density_fit_parameters
        self.density_fit_method = density_fit_method
        self.temperature_fit_parameters = temperature_fit_parameters
        self.temperature_fit_method = temperature_fit_method
        self.interp_order_magnetic_data = self._check_interp_order("magnetic", interp_order)
        self.interp_order_ne_data = self._check_interp_order("ne", interp_order) # TO REMOVE -- is this needed? or same interp_order for B and ne?
        self.interp_smoothing = interp_smoothing

        self._unpack_kwargs_for_circular_flux_surfaces(extras = kwargs)
        self._unpack_kwargs_for_plasma_launch(extras = kwargs)
        self._unpack_unused_kwargs(extras = kwargs)

    ##################################################
    #
    # PRIMARY INPUT CHECKS (for beam_me_up_3D)
    # For data passed at first call of beam_me_up_3D
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

    def _check_positive(self, name: str, value: Union[int, float]) -> Union[int, float]:
        if value <= 0: raise ValueError(f"`{name}` must be positive, but got {value}")
        else: return value
    
    def _check_path(self, name: Literal["magnetic", "ne", "Te", "output"], path: Union[str, Path]) -> Path:
        path = Path(path)
        if name in ["magnetic", "ne", "Te"] and not path.is_dir():
            raise ValueError(f"`{name}_data_path` must be a valid directory")
        elif name in ["output"] and not path.is_dir():
            print(f"`{name}_data_path` does not exist. Creating the folder now")
            os.makedirs(path)
        
        return path
    
    def _check_interp_order(self, name: Literal["magnetic", "ne"], order: Union[str, int]) -> Union[int, float]:
        _valid_interp_orders_int = {1: "linear", 3: "cubic", 5: "quintic"}
        _valid_interp_orders_str = dict((v,k) for k, v in _valid_interp_orders_int.items())
        _error = False

        if name == "magnetic":
            if   order in _valid_interp_orders_str: pass
            elif order in _valid_interp_orders_int: order = _valid_interp_orders_int[order]
            else: _error = True
        elif name == "ne":
            if   order in _valid_interp_orders_int: pass
            elif order in _valid_interp_orders_str: order = _valid_interp_orders_str[order]
            else: _error = True
        
        if _error: raise ValueError(f"""
        `interp_order` must be one of {list(_valid_interp_orders_int) + list(_valid_interp_orders_str)}, but got {order}
        
        Ignoring user-specified argument and setting `interp_order` = `quintic`. This may or may not cause issues
        """)
                
        return order
    
    def _unpack_kwargs_for_circular_flux_surfaces(self, **extras):
        B_T_axis       = extras.pop("B_T_axis", None)
        B_p_a          = extras.pop("B_p_a", None)
        R_axis         = extras.pop("R_axis", None)
        minor_radius_a = extras.pop("minor_radius_a", None)
        
        # TO REMOVE -- do we do it like this?
        # either all of them are defined, or none of them are
        # should also insert type hinting and checks
        _sum = bool(B_T_axis) + bool(B_p_a) + bool(R_axis) + bool(minor_radius_a)
        if _sum not in [0, 4]: raise ValueError(f"""
    For circular flux surfaces, all variables must be specified or None, but got:
       - B_T_axis = {B_T_axis}
       - B_p_a = {B_p_a}
       - R_axis = {R_axis}
       - minor_radius_a = {minor_radius_a}
    """)
        elif _sum == 4: self.circular_flux_surfaces_flag = True
        else:           self.circular_flux_surfaces_flag = False
        
        self.B_T_axis = B_T_axis
        self.B_p_a = B_p_a
        self.R_axis = R_axis
        self.minor_radius_a = minor_radius_a
        
    def _unpack_kwargs_for_plasma_launch(self, **extras):
        K_plasmaLaunch_cartesian = extras.pop("plasmaLaunch_K_cartesian", np.zeros(3))
        Psi_3D_plasmaLaunch_labframe_cartesian = extras.pop("plasmaLaunch_Psi_3D_lab_cartesian", np.zeros([3,3]))

        if isinstance(K_plasmaLaunch_cartesian, list): K_plasmaLaunch_cartesian = np.array(K_plasmaLaunch_cartesian)
        if isinstance(Psi_3D_plasmaLaunch_labframe_cartesian, list): Psi_3D_plasmaLaunch_labframe_cartesian = np.array(Psi_3D_plasmaLaunch_labframe_cartesian)

        _sum = np.sum(K_plasmaLaunch_cartesian) + np.sum(Psi_3D_plasmaLaunch_labframe_cartesian)

        if not self.vacuumLaunch_flag and _sum == 0: raise ValueError(f"""
    For plasma launch, `plasmaLaunch_K_cartesian` and `plasmaLaunch_Psi_3D_lab_cartesian`
    must be specified if `vacuumLaunch_flag` = False, but got:
       - vacuumLaunch_flag = {self.vacuumLaunch_flag}
       - plasmaLaunch_K_cartesian = {K_plasmaLaunch_cartesian}
       - plasmaLaunch_Psi_3D_lab_cartesian = {Psi_3D_plasmaLaunch_labframe_cartesian}
    """)
        elif self.vacuumLaunch_flag and _sum != 0: raise ValueError(f"""
    For vacuum launch, `plasmaLaunch_K_cartesian` and `plasmaLaunch_Psi_3D_lab_cartesian`
    must not be specified if `vacuumLaunch_flag` = True, but got:
       - vacuumLaunch_flag = {self.vacuumLaunch_flag}
       - plasmaLaunch_K_cartesian = {K_plasmaLaunch_cartesian}
       - plasmaLaunch_Psi_3D_lab_cartesian = {Psi_3D_plasmaLaunch_labframe_cartesian}
    """)
        
        self.K_plasmaLaunch_cartesian = K_plasmaLaunch_cartesian
        self.Psi_3D_plasmaLaunch_labframe_cartesian = Psi_3D_plasmaLaunch_labframe_cartesian
    
    def _unpack_unused_kwargs(self, extras):
        if extras:
            _indent = max(len(max(extras, key=len)), len("keyword")) + 3
            _printmsg = "\n".join(f"       - {k:<{_indent}} {v}" for k, v in extras.items())
            
            raise ValueError(f"""
    There are unsued keyword arguments that do not correspond to any function:
         {"keyword":<{_indent}} value \n{_printmsg}
    """)
    
    ##################################################
    #
    # ADDITIONAL FUNCTIONS
    #
    ##################################################
    
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
# SECONDARY INPUT CHECKS (for beam_me_up_3D)
# For data to be checked right before ray/beam tracing
#
##################################################

def _check_launch_position(poloidal_flux_enter: float, launch_position: FloatArray, field: MagneticField_3D_Cartesian) -> None:
    X, Y, Z = launch_position
    launch_psi = field.poloidal_flux(X, Y, Z)
    if launch_psi < poloidal_flux_enter:
        raise ValueError("Launch position (X={X:.4f}, Y={Y:.4f}, Z={Z:.4f}, psi={launch_psi:.4f}) is inside plasma (psi={poloidal_flux_enter})")

def check_input_before_ray_tracing(params: Parameters):
    log.info(f"Checking the validity of user-specified arguments")

    # Temporarily removing this function as the behaviour it checks for
    # is allowed in the new version which has poloidal_flux_enter and poloidal_zero_density
    # as separate input arguments
    # _check_launch_position(poloidal_flux_enter, launch_position, field)