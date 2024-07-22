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
        "density_fit_parameters": np.array([4.0, 1.0]),
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
        [1.9159087193750617, -0.06388797500308252, -0.0004997835782251143],
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
        "density_fit_parameters": np.array([4.0, 1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": True,
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
        [1.959548340506553, 0.1057637625447584, 0.00011320064162032355],
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
        "density_fit_parameters": np.array([4.0, 1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": True,
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
        [1.9186799430609414, 0.15549131819720569, 0.0016313326245700687],
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
        "density_fit_parameters": np.array([4.0, 1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": True,
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
        [1.8948392652387762, 0.023755528514813423, 1.5055640111241806e-05],
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
        "density_fit_parameters": np.array([4.0, 1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": True,
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
        [1.8954314162145214, -0.44718748795975066, -0.0020475610663150405],
        rtol=1e-2,
        atol=1e-2,
    )


def test_fund_6(tmp_path):
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": 4,
        "toroidal_launch_angle_Torbeam": 0,
        "launch_freq_GHz": 50,
        "mode_flag": -1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([0.8, 0, 0.2]),
        "density_fit_parameters": np.array([4.0, 1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": True,
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
        [1.0978655753712652, 0.06840555151937941, 3.1413690049921548],
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
        "density_fit_parameters": np.array([4.0, 1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": True,
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
        [1.9248542571474876, -0.01608339652600115, -4.952006410054158e-05],
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
        "density_fit_parameters": np.array([4.0, 1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": True,
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
        [1.8402531279851815, -0.17260313701585342, -0.0024810338010890884],
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
        "density_fit_parameters": np.array([4.0, 1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": True,
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
        [1.800610991524015, -0.20374718813163345, 0.0253947790819785],
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
        "density_fit_parameters": np.array([4.0, 1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": True,
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
        [1.403227523610646, -0.4181774880054182, -2.9500966593848186],
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
        "density_fit_parameters": np.array([4.0, 1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": True,
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
        [1.796484769128003, -0.09972777433924115, 3.1479306578685664],
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
        "density_fit_parameters": np.array([4.0, 1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": True,
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
        [1.9246718991949834, -0.13835603389402745, 0.0005892868969585501],
        rtol=1e-2,
        atol=1e-2,
    )
