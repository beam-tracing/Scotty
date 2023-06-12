# -*- coding: utf-8 -*-
# Copyright 2021 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""Initialisation.

This file contains the default settings for a range of various cases. This is
split into a set of dictionaries (`DEFAULT_DIAGNOSTIC_PARAMETERS`,
`LAUNCH_BEAM_METHODS`, and `DENSITY_FIT_PARAMETERS`) and functions
(`get_parameters_for_Scotty`, `beam_settings`, `ne_settings`, and
`user_settings`) that look up and parse the requested case.

"""
from __future__ import annotations
import math
import numpy as np
from numpy.typing import ArrayLike
import pathlib
from typing import Callable, Dict, NamedTuple, Optional, Any
import warnings

from .hornpy import make_my_horn
from .lensalot import make_my_lens
from .fun_general import (
    propagate_beam,
    propagate_circular_beam,
    find_nearest,
    genray_angles_from_mirror_angles,
    freq_GHz_to_wavenumber,
)
from .profile_fit import QuadraticFit, TanhFit, PolynomialFit, ProfileFit


def get_parameters_for_Scotty(
    diagnostic,
    launch_freq_GHz: Optional[float] = None,
    mirror_rotation: Optional[float] = None,
    mirror_tilt: Optional[float] = None,
    find_B_method: Optional[str] = None,
    find_ne_method: Optional[str] = None,
    equil_time: Optional[float] = None,
    shot: Optional[int] = None,
    user: Optional[str] = None,
):
    """Return default settings and parameters for the given diagnostic

    Arguments
    =========
    diagnostic:
        Name of the diagnostic to load parameters for. One of:

        - ``"DBS_NSTX_MAST"``
            - Doppler reflectometry (Neal Crocker, Jon Hillesheim, Tony Peebles)
            - Used on MAST, was on loan from NSTX

        - ``"DBS_SWIP_MAST-U"``
            - Doppler reflectometry (Peng Shi)

        - ``"DBS_UCLA_MAST-U"``
        - ``"CPS_UCLA_MAST-U"``
            - This system can either be used in CPS or DBS mode, but not both
              simultaneously (not yet, anyway)
            - CPS version not yet implemented
        - ``"hiK_Strath_MAST-U"``
            - High-k scattering diagnostic, Strathclyde (David Speirs, Kevin Ronald)
            - Not yet implemented
        - ``"DBS_synthetic"``
            - Circular flux surfaces
    launch_freq_GHz:
        Beam launch frequency in GHz
    mirror_rotation:
        Angle in degrees (FIXME: clarify)
    mirror_tilt:
        Angle in degrees (FIXME: clarify)
    find_B_method:
        Equilibrium magnetic field method. One of:

        - ``"torbeam"``: Loads data from ne.dat and topfile. I guess I
          should implement loading for inbeam.dat at some point, too
        - ``"UDA"``: Loads EFIT data directly from uda (not yet implemented)
        - ``"EFITpp"``: Uses MSE constrained EFIT
        - ``"UDA_saved"``: Loads EFIT data from file. UDA data must
          first be saved to said file

    find_ne_method:
        Density profile method. One of:

        - ``"torbeam"``
        - ``"EFITpp"``
        - ``"UDA_saved"``
    equil_time:
        Time in seconds
    shot:
        Shot number
    user:
        User profile settings
    """
    # Initialise dictionaries
    parameters: Dict[str, Any] = {
        "poloidal_launch_angle_Torbeam": None,
        "toroidal_launch_angle_Torbeam": None,
        "launch_freq_GHz": launch_freq_GHz,
        "mode_flag": None,
        "launch_beam_width": None,
        "launch_beam_curvature": None,
        "launch_position": None,
    }

    # User settings. Currently just loads paths
    ne_path, topfile_path, inbeam_path, efitpp_path, UDA_saved_path = user_settings(
        diagnostic, user, shot
    )

    # Assign keys that we already have
    if find_B_method is not None:
        parameters["find_B_method"] = find_B_method

    if equil_time is not None:
        parameters["equil_time"] = equil_time

    if shot is not None:
        parameters["shot"] = shot

    # Default settings for each diagnostic
    try:
        default_parameters = DEFAULT_DIAGNOSTIC_PARAMETERS[diagnostic]
    except KeyError:
        raise ValueError(
            f"Unknown diagnostic '{diagnostic}', available kinds are: {DEFAULT_DIAGNOSTIC_PARAMETERS.keys()}"
        )

    parameters.update(default_parameters(launch_freq_GHz))

    # Convert mirror angles to launch angles
    if (mirror_rotation is not None) and (mirror_tilt is not None):
        offset = np.rad2deg(math.atan2(125, 2432))
        toroidal_launch_angle, poloidal_launch_angle = genray_angles_from_mirror_angles(
            mirror_rotation, mirror_tilt, offset_for_window_norm_to_R=offset
        )

        parameters["poloidal_launch_angle_Torbeam"] = -poloidal_launch_angle
        parameters["toroidal_launch_angle_Torbeam"] = -toroidal_launch_angle

    # Density settings
    if find_B_method == "torbeam":
        parameters["ne_data_path"] = ne_path
        parameters["magnetic_data_path"] = topfile_path
    elif find_B_method in ["EFITpp", "UDA_saved", "test"]:
        parameters["ne_data_path"] = UDA_saved_path

        density_fit = ne_settings(diagnostic, shot, equil_time, find_ne_method)
        parameters["density_fit_method"] = density_fit.fit
        parameters[
            "poloidal_flux_zero_density"
        ] = density_fit.poloidal_flux_zero_density

    # MAST and MAST-U specific B-field and poloidal flux settings
    if find_B_method == "UDA":
        raise NotImplementedError
    elif find_B_method == "EFITpp":
        parameters["magnetic_data_path"] = efitpp_path
    elif find_B_method == "UDA_saved":
        parameters["magnetic_data_path"] = UDA_saved_path

    return parameters


class LaunchBeamParameters(NamedTuple):
    width: float
    """Width of the beam"""
    curvature: float
    """Curvature of the beam"""


def beam_settings(
    diagnostic: str, launch_freq_GHz: float, method: str = "data"
) -> LaunchBeamParameters:
    """Return the launch beam width and curvature

    Arguments
    =========
    diagnostic:
        Name of the diagnostic
    launch_freq_GHz:
        Frequency of the launch beam in GHz
    method:
        One of the following:

        - ``"horn_and_lens"``: Uses information about the horn and lens to
          calculate the launch beam properties
        - ``"thin_lens"``: Uses the thin lens approximation to calculate launch
          beam properties
        - ``"data"``: Uses pre-computed values
        - ``"estimate_var_w0"``: Estimate beam properties using
          frequency-dependent beam waist
        - ``"estimate_fix_w0"``: Estimate beam properties using
          frequency-independent beam waist


    .. note:: Not all methods are available for all diagnostics
    """

    try:
        diagnostic_methods = LAUNCH_BEAM_METHODS[diagnostic]
    except KeyError:
        raise ValueError(
            f"No launch beam settings for '{diagnostic}', available diagnostics are: "
            f"{LAUNCH_BEAM_METHODS.keys()}"
        )

    try:
        selected_method = diagnostic_methods[method]
    except KeyError:
        raise ValueError(
            f"No launch beam settings for method '{method}' on diagnostic "
            f"'{diagnostic}'. Available methods are: {diagnostic_methods.keys()}"
        )

    return LaunchBeamParameters(*selected_method(launch_freq_GHz))


class DensityFitParameters(NamedTuple):
    """Parameterised density"""

    fit: Optional[ProfileFit]
    """Fit parameterisation"""
    time_ms: Optional[float]
    """Actual shot time (in milliseconds) that parameters correspond to"""
    poloidal_flux_zero_density: Optional[float]
    """Poloidal flux surface label where the density goes to zero"""


def ne_settings(
    diagnostic: str,
    shot: Optional[int],
    time: Optional[float],
    find_ne_method: Optional[str],
) -> DensityFitParameters:
    """Get pre-existing density fit parameters from `DENSITY_FIT_PARAMETERS`"""

    if find_ne_method is None or shot is None or time is None:
        return DensityFitParameters(None, None, None)

    try:
        diagnostic_parameters = DENSITY_FIT_PARAMETERS[diagnostic]
    except KeyError:
        raise ValueError(
            f"No density fit data for diagnostic '{diagnostic}'. "
            f"Known diagnostics: {DENSITY_FIT_PARAMETERS.keys()}"
        )

    try:
        shot_parameters = diagnostic_parameters[shot]
    except KeyError:
        raise ValueError(
            f"No density fit data saved for shot {shot} and diagnostic '{diagnostic}'. "
            f"Available shots: {diagnostic_parameters.keys()}"
        )

    try:
        method_parameters = shot_parameters[find_ne_method]
    except KeyError:
        raise ValueError(
            f"No density fit data for method '{find_ne_method}' "
            f"(diagnostic: '{diagnostic}', shot: {shot}). "
            f"Available methods: {shot_parameters.keys()}"
        )

    ne_fits = method_parameters["fit"]
    ne_fit_times = method_parameters["time_ms"]

    nearest_time_idx = find_nearest(ne_fit_times, time)

    ne_fit = ne_fits[nearest_time_idx]
    ne_fit_time = ne_fit_times[nearest_time_idx]
    print("Nearest ne fit time:", ne_fit_time)

    return DensityFitParameters(ne_fit, ne_fit_time, ne_fit.poloidal_flux_zero_profile)


def user_settings(diagnostic, user, shot):
    """
    Choosing paths appropriately
    """

    # Default path: all input files in current working directory
    default_input_files_path = pathlib.Path(".")

    #########################
    # Initialising default paths
    # Paths are overwritten if specific users are chosen
    ne_path = default_input_files_path
    topfile_path = default_input_files_path
    inbeam_path = default_input_files_path
    efitpp_path = default_input_files_path
    UDA_saved_path = default_input_files_path
    #########################

    if user == "Freia":
        # Not yet properly implemented
        efitpp_path = None

    elif user in ["Valerian_desktop", "Valerian_laptop"]:
        if user == "Valerian_desktop":
            prefix = pathlib.Path("D:\\Dropbox\\")
        elif user == "Valerian_laptop":
            prefix = pathlib.Path("C:\\Dropbox\\")

        if diagnostic in ["DBS_NSTX_MAST", "DBS_SWIP_MAST-U", "DBS_UCLA_MAST-U"]:
            if shot in [29684]:
                # MAST reruns of EFIT. Done by Lucy Kogan.
                # 29684: no MSE data, but reprocessed with more constraints,
                # only good at the edge
                efitpp_path = (
                    prefix
                    / f"VHChen2020/Data/Equilibrium/MAST/Lucy_EFIT_runs/{shot}/epk_lkogan_01/"
                )
                ne_path = prefix / "VHChen2021/Data - Equilibrium/MAST/"

            elif shot in [30073, 30074, 30075, 30076, 30077]:
                # MAST reruns of EFIT. Done by Lucy Kogan.
                # 30073--30077: MSE data, processed better than original runs
                efitpp_path = (
                    prefix
                    / f"VHChen2020/Data/Equilibrium/MAST/Lucy_EFIT_runs/{shot}/epi_lkogan_01/"
                )
            elif shot in [29908]:
                # MAST EFIT runs. List of available shots not updated.
                efitpp_path = (
                    prefix
                    / f"VHChen2020/Data/Equilibrium/MAST/MSE_efitruns/{shot}/Pass0/"
                )
            elif shot in [45177]:
                # MAST EFIT runs. List of available shots not updated.
                efitpp_path = (
                    prefix
                    / f"VHChen2022/Data - Equilibrium/MSE_efitruns/{shot}/efit_sgibson_02/"
                )
            # If it's not any of the above shots, I'll assume that there's no efit++ data
            elif shot > 30471:  # MAST-U
                UDA_saved_path = (
                    prefix / "VHChen2020/Data/Equilibrium/MAST-U/Equilibrium_pyuda/"
                )
            else:
                UDA_saved_path = (
                    prefix / "VHChen2020/Data/Equilibrium/MAST/Equilibrium_pyuda/"
                )

        elif diagnostic == "DBS_UCLA_DIII-D_240":
            ne_path = prefix / "VHChen2021/Data - Equilibrium/DIII-D/"
            topfile_path = prefix / "VHChen2021/Data - Equilibrium/DIII-D/"

    return ne_path, topfile_path, inbeam_path, efitpp_path, UDA_saved_path


################################################################################
# Default parameters and parameterisations


def parameters_DBS_NSTX_MAST(launch_freq_GHz: float) -> dict:
    launch_beam = beam_settings("DBS_NSTX_MAST", launch_freq_GHz, method="data")

    return {
        # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition
        "launch_position": np.array([2.43521, 0, 0]),
        "launch_beam_width": launch_beam.width,
        "launch_beam_curvature": launch_beam.curvature,
        "Psi_BC_flag": "continuous",
        "figure_flag": True,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
    }


def parameters_DBS_UCLA_MAST_U(launch_freq_GHz: float) -> dict:
    print("Warning: launch_position is an estimate")
    # The z-position seems to vary from -9.7mm to -14.6mm, depending on launch angles
    launch_beam = beam_settings("DBS_UCLA_MAST-U", launch_freq_GHz, method="thin_lens")

    return {
        # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition
        # Launch position changes ~1mm based on lens settings
        "launch_position": np.array([2.278, 0, -0.01]),
        "launch_beam_width": launch_beam.width,
        "launch_beam_curvature": launch_beam.curvature,
        "Psi_BC_flag": "continuous",
        "figure_flag": True,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
    }


def parameters_DBS_SWIP_MAST_U(launch_freq_GHz: float) -> dict:
    print("Warning: launch_position is an estimate")
    launch_beam = beam_settings(
        "DBS_SWIP_MAST-U", launch_freq_GHz, method="estimate_fix_w0"
    )
    return {
        # Default settings
        # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition
        "launch_position": np.array([2.43521, 0, 0]),
        "launch_beam_width": launch_beam.width,
        "launch_beam_curvature": launch_beam.curvature,
        # I'm checking what this actually is from Peng. Currently using the
        # MAST UCLA DBS as a guide
        "Psi_BC_flag": "continuous",
        "figure_flag": True,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
    }


def parameters_DBS_UCLA_DIII_D_240(launch_freq_GHz: float) -> dict:
    launch_beam = beam_settings(
        "DBS_UCLA_DIII-D_240", launch_freq_GHz, method="thin_lens"
    )
    return {
        # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition
        "launch_position": np.array([2.587, 0, -0.0157]),
        "launch_beam_width": launch_beam.width,
        "launch_beam_curvature": launch_beam.curvature,
        "Psi_BC_flag": "continuous",
        "figure_flag": True,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
    }


def parameters_DBS_synthetic(launch_freq_GHz: float) -> dict:
    poloidal_flux_zero_density = 1.0
    poloidal_flux_zero_temperature = 1.0
    ne_fit = QuadraticFit(poloidal_flux_zero_density, 4.0)
    return {
        "poloidal_launch_angle_Torbeam": 6.0,
        "toroidal_launch_angle_Torbeam": 0.0,
        "launch_freq_GHz": 55.0,
        "mode_flag": 1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": 1 / -4.0,
        # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition
        "launch_position": np.array([2.587, 0, -0.0157]),
        "density_fit_method": ne_fit,
        "find_B_method": "analytical",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
        "poloidal_flux_zero_density": poloidal_flux_zero_density,
        "poloidal_flux_enter": 1.0,
        "B_T_axis": 1.0,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
        # Arguments for testing relativistic corrections
        "relativistic_flag": False,
        "poloidal_flux_zero_temperature": poloidal_flux_zero_temperature,
        "temperature_fit_method": None,
    }


DEFAULT_DIAGNOSTIC_PARAMETERS: Dict[str, Callable[[float], dict]] = {
    "DBS_NSTX_MAST": parameters_DBS_NSTX_MAST,
    "DBS_UCLA_MAST-U": parameters_DBS_UCLA_MAST_U,
    "DBS_SWIP_MAST-U": parameters_DBS_SWIP_MAST_U,
    "DBS_UCLA_DIII-D_240": parameters_DBS_UCLA_DIII_D_240,
    "DBS_synthetic": parameters_DBS_synthetic,
}
"""Functions that return default parameters for the corresponding diagnostic
"""


def launch_beam_DBS_NSTX_MAST_horn_and_lens(launch_freq_GHz):
    if launch_freq_GHz > 52.5:
        name = "MAST_V_band"
        horn_to_lens = 0.139  # V Band
        lens_to_mirror = 0.644  # lens to steering mirror

    elif launch_freq_GHz < 52.5:
        name = "MAST_Q_band"
        horn_to_lens = 0.270  # Q Band
        lens_to_mirror = 0.6425  # lens to steering mirror

    myLens = make_my_lens(name)
    myHorn = make_my_horn(name)
    wavenumber_K0 = freq_GHz_to_wavenumber(launch_freq_GHz)

    horn_width, horn_curvature = myHorn.output_beam(launch_freq_GHz)

    Psi_w_horn_cartersian = np.array(
        [
            [wavenumber_K0 * horn_curvature + 2j / (horn_width**2), 0],
            [0, wavenumber_K0 * horn_curvature + 2j / (horn_width**2)],
        ]
    )

    Psi_w_lens_cartesian_input = propagate_beam(
        Psi_w_horn_cartersian, horn_to_lens, launch_freq_GHz
    )

    Psi_w_lens_cartesian_output = myLens.output_beam(
        Psi_w_lens_cartesian_input, launch_freq_GHz
    )

    Psi_w_cartesian_launch = propagate_beam(
        Psi_w_lens_cartesian_output, lens_to_mirror, launch_freq_GHz
    )
    launch_beam_width = np.sqrt(2 / np.imag(Psi_w_cartesian_launch[0, 0]))
    launch_beam_curvature = np.real(Psi_w_cartesian_launch[0, 0]) / wavenumber_K0

    return launch_beam_width, launch_beam_curvature


def launch_beam_DBS_NSTX_MAST_data(launch_freq_GHz):
    data = np.array(
        [
            (30.0, 46.90319593e-3, -9211.13447598e-3),
            (32.5, 44.8730752e-3, -5327.42027113e-3),
            (35.0, 43.03016639e-3, -3834.26164617e-3),
            (37.5, 41.40562031e-3, -2902.09214589e-3),
            (42.5, 38.50759751e-3, -1961.58420391e-3),
            (45.0, 37.65323989e-3, -1636.82546574e-3),
            (47.5, 36.80672175e-3, -1432.59817651e-3),
            (50.0, 36.29814335e-3, -1296.20353095e-3),
            (55.0, 38.43065497e-3, -1437.24234181e-3),
            (57.5, 37.00251598e-3, -1549.7853604e-3),
            (60.0, 35.72544826e-3, -1683.5681014e-3),
            (62.5, 34.57900305e-3, -1843.91364265e-3),
            (67.5, 32.61150219e-3, -2277.59660009e-3),
            (70.0, 31.76347845e-3, -2577.93648944e-3),
            (72.5, 30.99132929e-3, -2964.57675092e-3),
            (75.0, 30.28611839e-3, -3479.14501841e-3),
        ],
        dtype=[
            ("freqs", np.float64),
            ("widths", np.float64),
            ("curvature", np.float64),
        ],
    )

    freq_idx = find_nearest(data["freqs"], launch_freq_GHz)
    launch_beam_width = data["widths"][freq_idx]
    launch_beam_curvature = 1 / data["curvature"][freq_idx]
    return launch_beam_width, launch_beam_curvature


def launch_beam_DBS_CIEMAT_JT60A_data(launch_freq_GHz):
    # 90 GHz
    launch_beam_width = 0.06323503329291348
    launch_beam_curvature = -0.5535179506038995
    return launch_beam_width, launch_beam_curvature


def launch_beam_DBS_UCLA_DIII_D_240_thin_lens(launch_freq_GHz):
    # The lens is directly in front of the waveguide (wg)
    wg_width = 0.025
    wg_curvature = 0.0
    wg_to_mirror = 0.08255

    # fmt: off
    data = np.array(
        [
            (50.0, 0.45181379), (50.5, 0.46384668),
            (51.0, 0.47595193), (51.5, 0.488133),
            (52.0, 0.50039298), (52.5, 0.51273458),
            (53.0, 0.52516023), (53.5, 0.53767209),
            (54.0, 0.55027209), (54.5, 0.56296199),
            (55.0, 0.57574337), (55.5, 0.58861765),
            (56.0, 0.60158613), (56.5, 0.61465001),
            (57.0, 0.62781035), (57.5, 0.64106815),
            (58.0, 0.65442432), (58.5, 0.66787969),
            (59.0, 0.68143503), (59.5, 0.69509106),
            (60.0, 0.70884843), (60.5, 0.72270775),
            (61.0, 0.73666959), (61.5, 0.75073448),
            (62.0, 0.7649029),  (62.5, 0.7791753),
            (63.0, 0.79355212), (63.5, 0.80803376),
            (64.0, 0.82262058), (64.5, 0.83731294),
            (65.0, 0.85211116), (65.5, 0.86701555),
            (66.0, 0.8820264),  (66.5, 0.89714399),
            (67.0, 0.91236857), (67.5, 0.92770038),
            (68.0, 0.94313965), (68.5, 0.95868659),
            (69.0, 0.97434142), (69.5, 0.99010431),
            (70.0, 1.00597547), (70.5, 1.02195505),
            (71.0, 1.03804323), (71.5, 1.05424016),
            (72.0, 1.07054599), (72.5, 1.08696085),
            (73.0, 1.10348489), (73.5, 1.12011823),
            (74.0, 1.13686099), (74.5, 1.15371329),
            (75.0, 1.17067524), (75.5, 1.18774694),
            (76.0, 1.20492849), (76.5, 1.22222),
            (77.0, 1.23962155), (77.5, 1.25713322),
            (78.0, 1.27475512), (78.5, 1.2924873),
            (79.0, 1.31032986), (79.5, 1.32828286),
            (80.0, 1.34634637),
        ],
        dtype=[("freqs_GHz", np.float64), ("focal_length", np.float64)],
    )
    # fmt: on
    nearest_freq_idx = find_nearest(data["freqs_GHz"], launch_freq_GHz)
    focal_length = data["focal_length"][nearest_freq_idx]

    myLens = make_my_lens(
        name="DBS_UCLA_DIII-D_240", lens_type="thin", focal_length=focal_length
    )
    wavenumber_K0 = freq_GHz_to_wavenumber(launch_freq_GHz)

    Psi_w_wg = np.array(
        [
            [wavenumber_K0 * wg_curvature + 2j / (wg_width**2), 0],
            [0, wavenumber_K0 * wg_curvature + 2j / (wg_width**2)],
        ]
    )

    Psi_w_lens = myLens.output_beam(Psi_w_wg, launch_freq_GHz)
    Psi_w_mirror = propagate_beam(Psi_w_lens, wg_to_mirror, launch_freq_GHz)

    launch_beam_width = np.sqrt(2 / np.imag(Psi_w_mirror[0, 0]))
    launch_beam_curvature = np.real(Psi_w_mirror[0, 0]) / wavenumber_K0
    return launch_beam_width, launch_beam_curvature


def launch_beam_DBS_UCLA_MAST_U_thin_lens(launch_freq_GHz):
    warnings.warn(
        "WARNING: DBS_UCLA_MAST-U lens known to change output beam properties "
        "depending on its y position, ignoring this effect"
    )

    myLens = make_my_lens("DBS_UCLA_MAST-U")
    wavenumber_K0 = freq_GHz_to_wavenumber(launch_freq_GHz)

    horn_width = 0.0064
    horn_curvature = 0.0
    horn_to_lens = 0.165

    wavenumber = freq_GHz_to_wavenumber(launch_freq_GHz)

    Psi_w_horn_cartersian = np.array(
        [
            [wavenumber * horn_curvature + 2j * horn_width ** (-2), 0],
            [0, wavenumber * horn_curvature + 2j * horn_width ** (-2)],
        ]
    )

    Psi_w_lens_cartesian_input = propagate_beam(
        Psi_w_horn_cartersian, horn_to_lens, launch_freq_GHz
    )

    Psi_w_lens_cartesian_output = myLens.output_beam(
        Psi_w_lens_cartesian_input, launch_freq_GHz
    )
    launch_beam_width = np.sqrt(2 / np.imag(Psi_w_lens_cartesian_output[0, 0]))
    launch_beam_curvature = np.real(Psi_w_lens_cartesian_output[0, 0]) / wavenumber_K0
    return launch_beam_width, launch_beam_curvature


def launch_beam_DBS_SWIP_MAST_U_estimate_var_w0(launch_freq_GHz):
    if launch_freq_GHz <= 50.0:  # Q band
        w0 = np.sqrt(launch_freq_GHz / 40) * 0.08
    else:  # V band
        w0 = np.sqrt(launch_freq_GHz / 60) * 0.04

    # window to steering mirror, negative because the mirror is behind the window
    distance = -0.277
    wavenumber_K0 = freq_GHz_to_wavenumber(launch_freq_GHz)
    return propagate_circular_beam(distance, wavenumber_K0, w0, launch_freq_GHz)


def launch_beam_DBS_SWIP_MAST_U_estimate_fix_w0(launch_freq_GHz):
    if launch_freq_GHz <= 50.0:  # Q band
        w0 = 0.08
    else:  # V band
        w0 = 0.04

    # window to steering mirror, negative because the mirror is behind the window
    distance = -0.277
    wavenumber_K0 = freq_GHz_to_wavenumber(launch_freq_GHz)
    return propagate_circular_beam(distance, wavenumber_K0, w0, launch_freq_GHz)


LAUNCH_BEAM_METHODS: Dict[str, Dict[str, Callable[[float], LaunchBeamParameters]]] = {
    "DBS_NSTX_MAST": {
        "horn_and_lens": launch_beam_DBS_NSTX_MAST_horn_and_lens,
        "data": launch_beam_DBS_NSTX_MAST_data,
    },
    "DBS_CIEMAT_JT60SA": {"data": launch_beam_DBS_CIEMAT_JT60A_data},
    "DBS_UCLA_DIII-D_240": {"thin_lens": launch_beam_DBS_UCLA_DIII_D_240_thin_lens},
    "DBS_UCLA_MAST-U": {"thin_lens": launch_beam_DBS_UCLA_MAST_U_thin_lens},
    "DBS_SWIP_MAST-U": {
        "estimate_var_w0": launch_beam_DBS_SWIP_MAST_U_estimate_var_w0,
        "estimate_fix_w0": launch_beam_DBS_SWIP_MAST_U_estimate_fix_w0,
    },
}
"""Functions that return launch beam parameters for the corresponding diagnostic"""


fit_dtype = [("time_ms", np.float64), ("fit", ProfileFit)]

DENSITY_FIT_PARAMETERS = {
    "DBS_NSTX_MAST": {
        29684: {
            "tanh": np.array(
                [
                    (0.167, TanhFit(1.2, 2.1, -1.9)),
                    (0.179, TanhFit(1.25, 2.1, -1.9)),
                    (0.192, TanhFit(1.2, 2.3, -1.9)),
                    (0.200, TanhFit(1.2, 2.4, -2.0)),
                    (0.217, TanhFit(1.2, 2.5, -2.1)),
                ],
                dtype=fit_dtype,
            ),
        },
        29908: {
            "tanh": np.array(
                [
                    (0.150, TanhFit(1.18, 2.3, -1.9)),
                    (0.160, TanhFit(1.15, 2.55, -2.2)),
                    (0.170, TanhFit(1.15, 2.8, -2.2)),
                    (0.180, TanhFit(1.2, 3.0, -2.35)),
                    (0.190, TanhFit(1.22, 3.25, -2.4)),
                    (0.200, TanhFit(1.15, 3.7, -2.7)),
                    (0.210, TanhFit(1.2, 4.2, -2.0)),
                    (0.220, TanhFit(1.24, 4.5, -1.8)),
                    (0.230, TanhFit(1.2, 4.8, -1.8)),
                    (0.240, TanhFit(1.2, 5.2, -1.8)),
                    (0.250, TanhFit(1.1, 5.2, -2.8)),
                    (0.260, TanhFit(1.15, 5.7, -1.9)),
                    (0.270, TanhFit(1.1, 5.8, -2.2)),
                    (0.280, TanhFit(1.15, 6.5, -1.7)),
                    (0.290, TanhFit(1.1, 6.6, -1.8)),
                ],
                dtype=fit_dtype,
            ),
            # fmt: off
            "poly3": np.array(
                [
                    (0.16, PolynomialFit(1.14908613, -3.39920666, 3.3767761, -1.55984715, 2.49116064)),
                    (0.17, PolynomialFit(1.13336773, -3.31670147, 2.24970438, -0.46971473, 2.47113803)),
                    (0.18, PolynomialFit(1.13850717, -3.44610169, 1.69591882, 0.22709583, 2.62872259)),
                    (0.19, PolynomialFit(1.1274871, -4.91157473, 3.12397459, 0.30956902, 2.71940548)),
                    (0.20, PolynomialFit(1.09851547, -6.99408536, 4.98094795, 0.36188724, 2.86325923)),
                    (0.21, PolynomialFit(1.1302204, -4.01026147, 0.89218099, 1.24564799, 3.24225355)),
                    (0.22, PolynomialFit(1.15828467, -4.21483706, 1.73180927, 0.63975703, 3.48532344)),
                    (0.23, PolynomialFit(1.14394922, -3.89636166, 1.00604874, 0.5745085, 3.85908854)),
                    (0.24, PolynomialFit(1.1153502, -4.75241047, 1.34810624, 0.7207821, 4.11300423)),
                    (0.25, PolynomialFit(1.13408871, -2.98806121, -1.48680106, 1.51977942, 4.54713015)),
                ],
                dtype=fit_dtype,
            ),
            # fmt: on
        },
        29980: {"tanh": np.array([(0.200, TanhFit(1.12, 2.3, -2.6))], dtype=fit_dtype)},
        30073: {
            # Fit underestimates TS density when polflux < 0.2 (roughly, for some of the times)
            "tanh": np.array(
                [
                    (0.190, TanhFit(1.1, 2.8, -1.4)),
                    (0.200, TanhFit(1.15, 2.9, -1.4)),
                    (0.210, TanhFit(1.2, 3.0, -1.3)),
                    (0.220, TanhFit(1.2, 3.4, -1.2)),
                    (0.230, TanhFit(1.2, 3.6, -1.2)),
                    (0.240, TanhFit(1.2, 4.0, -1.2)),
                    (0.250, TanhFit(1.2, 4.4, -1.2)),
                ],
                dtype=fit_dtype,
            ),
        },
        45091: {
            "tanh": np.array(
                [
                    (0.390, TanhFit(1.2, 5.5, -0.6)),
                    (0.400, TanhFit(1.2, 4.8, -0.6)),
                    (0.410, TanhFit(1.2, 3.0, -1.3)),
                ],
                dtype=fit_dtype,
            )
        },
        45154: {"tanh": np.array([(0.510, TanhFit(1.12, 2.4, -1.8))], dtype=fit_dtype)},
        45189: {
            "tanh": np.array(
                [(0.200, TanhFit(1.12, 2.3, -1.8)), (0.650, TanhFit(1.2, 3.5, -1.35))],
                dtype=fit_dtype,
            )
        },
    }
}
"""Density fitting parameters for diagnostic/shot combinations"""
