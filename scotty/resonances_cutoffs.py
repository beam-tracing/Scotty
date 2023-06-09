# -*- coding: utf-8 -*-
# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""
Functions for finding plasma and cyclotron frequencies, as well as X-mode cutoffs in terms of poloidal flux coordinates.
Functions ultimately should work with both relativistic corrections and no relativistic corrections.
For now, relativistic electron mass corrections are calculated with the Mazzucato corrections in find_electron_mass, though
this is not the correct way to calculate the relativistic frequency shift/broadening.
To be eventually updated with proper broadening calculations based on the temperature profile.
"""

from __future__ import annotations
import numpy as np
from scipy import constants as constants
import json
import pathlib

from scotty.fun_general import find_electron_mass, angular_frequency_to_freq_GHZ


from scotty.profile_fit import profile_fit, ProfileFitLike
from scotty.geometry import (
    MagneticField,
    CircularCrossSectionField,
    ConstantCurrentDensityField,
    InterpolatedField,
    CurvySlabField,
    EFITField,
)

from scotty.beam_me_up import (
    make_density_fit,
    make_temperature_fit,
    create_magnetic_geometry,
)

from typing import Optional, Union, Sequence


def resonances_cutoffs(
    relativistic_flag: bool = False,
    # Fitting parameters for profile
    find_B_method: Union[str, MagneticField] = "torbeam",
    density_fit_parameters: Optional[Sequence] = None,
    temperature_fit_parameters: Optional[Sequence] = None,
    shot=None,
    equil_time=None,
    poloidal_flux_zero_density: float = 1.0,  ## When polflux >= poloidal_flux_zero_density, Scotty sets density = 0
    poloidal_flux_zero_temperature: float = 1.0,
    density_fit_method: Optional[Union[str, ProfileFitLike]] = None,
    temperature_fit_method: Optional[Union[str, ProfileFitLike]] = None,
    # Input and Output Settings
    ne_data_path=pathlib.Path("."),
    magnetic_data_path=pathlib.Path("."),
    Te_data_path=pathlib.Path("."),
    output_path=pathlib.Path("."),
    input_filename_suffix="",
    B_T_axis=None,
    B_p_a=None,
    R_axis=None,
    minor_radius_a=None,
    # Finite-difference and interpolation parameters
    delta_R: float = -0.0001,  # in the same units as data_R_coord
    delta_Z: float = 0.0001,  # in the same units as data_Z_coord
    interp_order=5,  # For the 2D interpolation functions
    interp_smoothing=0,  # For the 2D interpolation functions. For no smoothing, set to 0
):
    r"""
    Function arguments similar to beam_me_up but does not take in arguments relating to beam launch parameters. Calculates resonances/cutoff
    frequencies and returns a nested dictionary of functions for plotting, categorised by 'profiles' and 'frequencies'. All functions are
    functions of radial distance in the midplane, with an option to alter the value from Z=0 (midplane).

    Keys for the output functions are ['profiles']['density'/'temperature'/'poloidal_flux'/'B_magnitude'] and
    ['frequencies']['plasma_frequency'/'cyclotron_frequency'/'X_mode_right_cutoff'/'X_mode_left_cutoff'].

    Frequency functions are in GHz.
    """

    print(f"Relativistic_flag is set to {relativistic_flag}.")
    print("Finding resonances and cutoffs...")

    # Ensure paths are `pathlib.Path`
    ne_data_path = pathlib.Path(ne_data_path)
    magnetic_data_path = pathlib.Path(magnetic_data_path)
    Te_data_path = pathlib.Path(Te_data_path)
    output_path = pathlib.Path(output_path)

    # Reconstruct density profile
    if density_fit_parameters is None and (
        density_fit_method in [None, "smoothing-spline-file"]
    ):
        ne_filename = ne_data_path / f"ne{input_filename_suffix}.dat"
        density_fit_parameters = [ne_filename, interp_order, interp_smoothing]

        # FIXME: Read data so it can be saved later
        # ne_data = np.fromfile(ne_filename, dtype=float, sep="   ")
        # ne_data_density_array = ne_data[2::2]
        # ne_data_radialcoord_array = ne_data[1::2]
    else:
        ne_filename = None

    electron_density_function = make_density_fit(
        density_fit_method,
        poloidal_flux_zero_density,
        density_fit_parameters,
        ne_filename,
    )

    # Reconstruct temperature profile
    Te_filename = None
    if relativistic_flag:
        if temperature_fit_parameters is None and (
            temperature_fit_method
            in [
                None,
                "smoothing-spline-file",
            ]
        ):
            Te_filename = Te_data_path / f"Te{input_filename_suffix}.dat"
            temperature_fit_parameters = [Te_filename, interp_order, interp_smoothing]

    if relativistic_flag:
        temperature_function = make_temperature_fit(
            temperature_fit_method,
            poloidal_flux_zero_temperature,
            temperature_fit_parameters,
            Te_filename,
        )
    else:
        temperature_function = null_temperature_function

    # Reconstruct B-field profile
    field = create_magnetic_geometry(
        find_B_method,
        magnetic_data_path,
        input_filename_suffix,
        interp_order,
        interp_smoothing,
        B_T_axis,
        R_axis,
        minor_radius_a,
        B_p_a,
        shot,
        equil_time,
        delta_R,
        delta_Z,
    )

    def density_R_Z(R_coord, Z_coord=0):
        poloidal_flux = field.poloidal_flux(R_coord, Z_coord)
        return electron_density_function(poloidal_flux)

    def temperature_R_Z(R_coord, Z_coord=0):
        poloidal_flux = field.poloidal_flux(R_coord, Z_coord)
        return temperature_function(poloidal_flux)

    def poloidal_flux_R_Z(R_coord, Z_coord=0):
        return field.poloidal_flux(R_coord, Z_coord)

    def B_Total_R_Z(R_coord, Z_coord=0):
        return np.sqrt(
            field.B_R(R_coord, Z_coord) ** 2
            + field.B_T(R_coord, Z_coord) ** 2
            + field.B_Z(R_coord, Z_coord) ** 2
        )

    def plasma_freq_function(R_coord, Z_coord=0):
        electron_density = density_R_Z(R_coord, Z_coord)
        temperature = temperature_R_Z(R_coord, Z_coord)
        electron_mass = find_electron_mass(temperature)
        plasma_freq = constants.e * np.sqrt(
            electron_density * 10**19 / (constants.epsilon_0 * electron_mass)
        )
        return plasma_freq

    def gyro_freq_function(R_coord, Z_coord=0):
        temperature = temperature_R_Z(R_coord, Z_coord)
        electron_mass = find_electron_mass(temperature)
        B_Total = B_Total_R_Z(R_coord, Z_coord)
        gyro_freq = constants.e * B_Total / electron_mass
        return gyro_freq

    def X_mode_R_cutoff_function(R_coord, Z_coord=0):
        return 0.5 * (
            gyro_freq_function(R_coord, Z_coord)
            + np.sqrt(
                gyro_freq_function(R_coord, Z_coord) ** 2
                + 4 * plasma_freq_function(R_coord, Z_coord) ** 2
            )
        )

    def X_mode_L_cutoff_function(R_coord, Z_coord=0):
        return 0.5 * (
            -gyro_freq_function(R_coord, Z_coord)
            + np.sqrt(
                gyro_freq_function(R_coord, Z_coord) ** 2
                + 4 * plasma_freq_function(R_coord, Z_coord) ** 2
            )
        )

    def convert_to_GHz(function):
        return lambda R_coord, Z_coord=0: angular_frequency_to_freq_GHZ(
            function(R_coord, Z_coord)
        )

    # def upper_hybrid_function(R_coord, Z_coord=0):
    #    return np.sqrt(plasma_freq_function(R_coord, Z_coord)**2 + gyro_freq_function(R_coord, Z_coord)**2)

    functions_dict = {
        "profiles": {
            "density": density_R_Z,
            "temperature": temperature_R_Z,
            "poloidal_flux": poloidal_flux_R_Z,
            "B_magnitude": B_Total_R_Z,
        },
        "frequencies": {
            "plasma_frequency": convert_to_GHz(plasma_freq_function),
            "cyclotron_frequency": convert_to_GHz(gyro_freq_function),
            "X_mode_right_cutoff": convert_to_GHz(X_mode_R_cutoff_function),
            "X_mode_left_cutoff": convert_to_GHz(X_mode_L_cutoff_function),
        },
    }
    return functions_dict


def null_temperature_function(*args):
    return None
