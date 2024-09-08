# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:06:33 2024

@author: matth
"""

from scotty.beam_me_up import (
    beam_me_up,
    create_magnetic_geometry,
    make_density_fit,
)
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.torbeam import Torbeam
from scotty.geometry import CircularCrossSectionField
from scotty.fun_general import freq_GHz_to_angular_frequency
from scotty.profile_fit import QuadraticFit
from scotty.hamiltonian import Hamiltonian
from scotty.launch import find_entry_point, launch_beam
from scotty.typing import FloatArray

import json
import numpy as np
from numpy.testing import assert_allclose
import pathlib
import pytest
import xarray as xr

# Print more of arrays in failed tests
np.set_printoptions(linewidth=120, threshold=100)


# Test_fund: test to see whether the beam will be absorbed when its frequency is equal
# to the fundamental electron cyclotron frequency.
# Six test cases were created to test for this by varying beam frequency, launch angles,
# launch positions (including launching from inboard side), polarisations and magnetic field strength.


def test_fund_1(tmp_path):
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": 3,
        "toroidal_launch_angle_Torbeam": 0,
        "launch_freq_GHz": 33,
        "mode_flag": 1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([3, 0, 0]),
        "density_fit_parameters": np.array([1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.5,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)["analysis"]
    assert_allclose(
        [
            float(output["q_R"][-1]),
            float(output["q_Z"][-1]),
            float(output["q_zeta"][-1]),
        ],
        [1.9162396978780787, -0.05765344712076549, -7.654831340237264e-05],
        rtol=1e-2,
        atol=1e-2,
    )


def test_fund_2(tmp_path):
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": -3,
        "toroidal_launch_angle_Torbeam": 0,
        "launch_freq_GHz": 28,
        "mode_flag": 1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([3, 0, 0.05]),
        "density_fit_parameters": np.array([1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.3,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)["analysis"]
    assert_allclose(
        [
            float(output["q_R"][-1]),
            float(output["q_Z"][-1]),
            float(output["q_zeta"][-1]),
        ],
        [1.9592789345742416, 0.10479082636734298, 2.669217726708174e-05],
        rtol=1e-2,
        atol=1e-2,
    )


def test_fund_3(tmp_path):
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": -5,
        "toroidal_launch_angle_Torbeam": 0,
        "launch_freq_GHz": 29,
        "mode_flag": 1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([3, 0, 0.05]),
        "density_fit_parameters": np.array([1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.3,
        "B_p_a": 0.2,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)["analysis"]
    assert_allclose(
        [
            float(output["q_R"][-1]),
            float(output["q_Z"][-1]),
            float(output["q_zeta"][-1]),
        ],
        [1.9188624402871175, 0.1459755482521055, 0.0002667820559604742],
        rtol=1e-2,
        atol=1e-2,
    )


def test_fund_4(tmp_path):
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": 3,
        "toroidal_launch_angle_Torbeam": 0,
        "launch_freq_GHz": 40,
        "mode_flag": 1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([2.4, 0, 0.05]),
        "density_fit_parameters": np.array([1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.8,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)["analysis"]
    assert_allclose(
        [
            float(output["q_R"][-1]),
            float(output["q_Z"][-1]),
            float(output["q_zeta"][-1]),
        ],
        [1.894834149051581, 0.023559742880371104, 2.6650767398267687e-06],
        rtol=1e-2,
        atol=1e-2,
    )


def test_fund_5(tmp_path):
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": 0,
        "toroidal_launch_angle_Torbeam": 0,
        "launch_freq_GHz": 40,
        "mode_flag": 1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([2.2, 0, -0.4]),
        "density_fit_parameters": np.array([1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.8,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.7,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)["analysis"]
    assert_allclose(
        [
            float(output["q_R"][-1]),
            float(output["q_Z"][-1]),
            float(output["q_zeta"][-1]),
        ],
        [1.8959753930261396, -0.4072190857298863, -0.0004257007543867974],
        rtol=1e-2,
        atol=1e-2,
    )


def test_fund_6(tmp_path):
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": -2,
        "toroidal_launch_angle_Torbeam": 0,
        "launch_freq_GHz": 50,
        "mode_flag": -1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([0.8, 0, 0.2]),
        "density_fit_parameters": np.array([1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.3,
        "B_p_a": 0.2,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)["analysis"]
    assert_allclose(
        [
            float(output["q_R"][-1]),
            float(output["q_Z"][-1]),
            float(output["q_zeta"][-1]),
        ],
        [1.0983935588516547, 0.26638074344861923, 3.1415778015244564],
        rtol=1e-2,
        atol=1e-2,
    )


# Test_sec_harm: test to see whether the beam will be absorbed when its frequency is equal
# to the second harmonic electron cyclotron frequency.
# Six test cases were created to test for this by varying beam frequency, launch angles,
# launch positions (including launching from inboard side), polarisations and magnetic field strength.


def test_sec_harm_1(tmp_path):
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": 0,
        "toroidal_launch_angle_Torbeam": 0.0,
        "launch_freq_GHz": 44,
        "mode_flag": 1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([2.587, 0.0, -0.0157]),
        "density_fit_parameters": np.array([1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.0,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)["analysis"]
    assert_allclose(
        [
            float(output["q_R"][-1]),
            float(output["q_Z"][-1]),
            float(output["q_zeta"][-1]),
        ],
        [1.9249442300308424, -0.015775382056218094, -1.0554438814446962e-05],
        rtol=1e-2,
        atol=1e-2,
    )


def test_sec_harm_2(tmp_path):
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": 10,
        "toroidal_launch_angle_Torbeam": 0.0,
        "launch_freq_GHz": 46,
        "mode_flag": 1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([2.587, 0.0, -0.0157]),
        "density_fit_parameters": np.array([1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.0,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)["analysis"]
    assert_allclose(
        [
            float(output["q_R"][-1]),
            float(output["q_Z"][-1]),
            float(output["q_zeta"][-1]),
        ],
        [1.8406422101660265, -0.15099269051484762, -0.00044930268212935675],
        rtol=1e-2,
        atol=1e-2,
    )


def test_sec_harm_3(tmp_path):
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": 10,
        "toroidal_launch_angle_Torbeam": -4,
        "launch_freq_GHz": 47,
        "mode_flag": 1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([2.587, 0.0, -0.0157]),
        "density_fit_parameters": np.array([1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.0,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)["analysis"]
    assert_allclose(
        [
            float(output["q_R"][-1]),
            float(output["q_Z"][-1]),
            float(output["q_zeta"][-1]),
        ],
        [1.8013024517505818, -0.16041011588294915, 0.029613857265963682],
        rtol=1e-2,
        atol=1e-2,
    )


def test_sec_harm_4(tmp_path):
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": 5,
        "toroidal_launch_angle_Torbeam": 10,
        "launch_freq_GHz": 60,
        "mode_flag": 1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([0.1, 0, -0.2]),
        "density_fit_parameters": np.array([1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.0,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)["analysis"]
    assert_allclose(
        [
            float(output["q_R"][-1]),
            float(output["q_Z"][-1]),
            float(output["q_zeta"][-1]),
        ],
        [1.4028531737131118, -0.34622305181939556, -2.9534075602992305],
        rtol=1e-2,
        atol=1e-2,
    )


def test_sec_harm_5(tmp_path):
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": 1,
        "toroidal_launch_angle_Torbeam": 0,
        "launch_freq_GHz": 70,
        "mode_flag": 1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([0.1, 0, 0]),
        "density_fit_parameters": np.array([1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.5,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)["analysis"]
    assert_allclose(
        [
            float(output["q_R"][-1]),
            float(output["q_Z"][-1]),
            float(output["q_zeta"][-1]),
        ],
        [1.7972393930018864, -0.04033405267167445, 3.142860330914468],
        rtol=1e-2,
        atol=1e-2,
    )


def test_sec_harm_6(tmp_path):
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": 10,
        "toroidal_launch_angle_Torbeam": 0.0,
        "launch_freq_GHz": 44,
        "mode_flag": -1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([2.587, 0.0, -0.0157]),
        "density_fit_parameters": np.array([1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.0,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)["analysis"]
    assert_allclose(
        [
            float(output["q_R"][-1]),
            float(output["q_Z"][-1]),
            float(output["q_zeta"][-1]),
        ],
        [1.9247988276944217, -0.1334200438972866, 0.00010847150538311593],
        rtol=1e-2,
        atol=1e-2,
    )
