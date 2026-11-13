"""Relativistic resonances and the resonance flags (beam-tracing/Scotty#151), ported to v4.

Generated from #151's v3 tests in the version of #161, whose expected end points are for
find_B_method="analytical" = CircularFluxSurfaceField, the field v4's "analytical" now gives too
(only the v3 -> v4 argument and output names are changed). From the v3 file:

The tests are verified by first plotting the absorption curves in the R-Z plane. Then, the trajectory
is plotted on this same plot. If the trajectory stops at the curve, it means that the function is
working. These tests are for circular flux surfaces. The *_rel tests repeat the six fundamental and six
second-harmonic cases of tests/test_resonance.py with a temperature profile, so that the resonances
include the relativistic mass correction; both low (< 5 keV) and high (> 5 keV) temperatures are used.
test_default_flags_stop, test_first_harmonic_raise and test_first_harmonic_stop_second_pass
check first_harmonic_flag, second_harmonic_flag and upper_hybrid_flag.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scotty.beam_me_up import beam_me_up
from scotty.fun_general import (
    find_normalised_gyro_ang_freq,
    freq_GHz_to_angular_frequency,
)

# Print more of arrays in failed tests
np.set_printoptions(linewidth=120, threshold=100)


def test_fund_1_rel(tmp_path):
    kwargs_dict = {
        "geometry": "cylindrical",
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
        "relativistic_flag": True,
        "temperature_fit_parameters": np.array([2.0]),  # Low temperature <5 keV
        "temperature_fit_method": "quadratic",
        "launch_flag": "vacuum_analytic",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "poloidal_flux_zero_temperature": 1,
        "B_T_axis": 1.5,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)
    assert_allclose(
        [
            float(output["solver/q_R_output"][-1]),
            float(output["solver/q_Z_output"][-1]),
            float(output["solver/q_zeta_output"][-1]),
        ],
        [1.9103564491962086, -0.05809777044206722, -8.882382142451245e-05],
        rtol=1e-2,
        atol=1e-2,
    )


def test_fund_2_rel(tmp_path):
    kwargs_dict = {
        "geometry": "cylindrical",
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
        "relativistic_flag": True,
        "temperature_fit_parameters": np.array([6.0]),  # high temperature >5 keV
        "temperature_fit_method": "quadratic",
        "find_B_method": "analytical",
        "launch_flag": "vacuum_analytic",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_temperature": 1,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.3,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }

    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)
    assert_allclose(
        [
            float(output["solver/q_R_output"][-1]),
            float(output["solver/q_Z_output"][-1]),
            float(output["solver/q_zeta_output"][-1]),
        ],
        [1.9514969586482378, 0.10535069088939782, 3.9809690264897525e-05],
        rtol=1e-2,
        atol=1e-2,
    )


def test_fund_3_rel(tmp_path):
    kwargs_dict = {
        "geometry": "cylindrical",
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
        "launch_flag": "vacuum_analytic",
        "Psi_BC_flag": "continuous",
        "relativistic_flag": True,
        "temperature_fit_parameters": np.array([10.0]),  # high temperature >5 keV
        "temperature_fit_method": "quadratic",
        "figure_flag": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.3,
        "B_p_a": 0.2,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)
    assert_allclose(
        [
            float(output["solver/q_R_output"][-1]),
            float(output["solver/q_Z_output"][-1]),
            float(output["solver/q_zeta_output"][-1]),
        ],
        [1.8345510858149414, 0.17316950433821163, 0.0023098532726736633],
        rtol=1e-2,
        atol=1e-2,
    )


def test_fund_4_rel(tmp_path):
    kwargs_dict = {
        "geometry": "cylindrical",
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
        "relativistic_flag": True,
        "temperature_fit_parameters": np.array([20.0]),  # high temperature >5 keV
        "temperature_fit_method": "quadratic",
        "launch_flag": "vacuum_analytic",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.8,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)
    assert_allclose(
        [
            float(output["solver/q_R_output"][-1]),
            float(output["solver/q_Z_output"][-1]),
            float(output["solver/q_zeta_output"][-1]),
        ],
        [1.7307657759972634, 0.015302520233810981, 1.7185443880748966e-05],
        rtol=1e-2,
        atol=1e-2,
    )


def test_fund_5_rel(tmp_path):
    kwargs_dict = {
        "geometry": "cylindrical",
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
        "relativistic_flag": True,
        "temperature_fit_parameters": np.array([3.5]),  # Low temperature <5 keV
        "temperature_fit_method": "quadratic",
        "launch_flag": "vacuum_analytic",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.8,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.7,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)
    assert_allclose(
        [
            float(output["solver/q_R_output"][-1]),
            float(output["solver/q_Z_output"][-1]),
            float(output["solver/q_zeta_output"][-1]),
        ],
        [1.8841856569285043, -0.4082171370330961, -0.000483268825027544],
        rtol=1e-2,
        atol=1e-2,
    )


def test_fund_6_rel(tmp_path):
    kwargs_dict = {
        "geometry": "cylindrical",
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
        "launch_flag": "vacuum_analytic",
        "Psi_BC_flag": "continuous",
        "relativistic_flag": True,
        "temperature_fit_parameters": np.array([6.7]),  # high temperature >5 keV
        "temperature_fit_method": "quadratic",
        "figure_flag": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.3,
        "B_p_a": 0.2,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)
    assert_allclose(
        [
            float(output["solver/q_R_output"][-1]),
            float(output["solver/q_Z_output"][-1]),
            float(output["solver/q_zeta_output"][-1]),
        ],
        [1.0957854232564677, 0.26627001769942255, 3.1415811409120136],
        rtol=1e-2,
        atol=1e-2,
    )


# Test_sec_harm: test to see whether the beam will be absorbed when its frequency is equal
# to the second harmonic electron cyclotron frequency.
# Six test cases were created to test for this by varying beam frequency, launch angles,
# launch positions (including launching from inboard side), polarisations and magnetic field strength.


# Test_sec_harm_X_rel: test to see whether the beam will be absorbed when its frequency is equal
# to the second harmonic electron cyclotron frequency, with relativistic corrections
# Same six test cases from test_sec_harm were used, but we include the temperature
# to add relativistic effects


def test_sec_harm_1_rel(tmp_path):
    kwargs_dict = {
        "geometry": "cylindrical",
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
        "launch_flag": "vacuum_analytic",
        "Psi_BC_flag": "continuous",
        "relativistic_flag": True,
        "temperature_fit_parameters": np.array([7.8]),  # high temperature >5 keV
        "temperature_fit_method": "quadratic",
        "figure_flag": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.0,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)
    assert_allclose(
        [
            float(output["solver/q_R_output"][-1]),
            float(output["solver/q_Z_output"][-1]),
            float(output["solver/q_zeta_output"][-1]),
        ],
        [1.8631895522670991, -0.016173229853573817, -4.274971098350256e-05],
        rtol=1e-2,
        atol=1e-2,
    )


def test_sec_harm_2_rel(tmp_path):
    kwargs_dict = {
        "geometry": "cylindrical",
        "poloidal_launch_angle_Torbeam": 10,
        "toroidal_launch_angle_Torbeam": 0.0,
        "launch_freq_GHz": 46,
        "mode_flag": 1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([2.587, 0.0, -0.0157]),
        "density_fit_parameters": np.array([1]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "launch_flag": "vacuum_analytic",
        "Psi_BC_flag": "continuous",
        "relativistic_flag": True,
        "temperature_fit_parameters": np.array([0.8]),  # Low temperature <5 keV
        "temperature_fit_method": "quadratic",
        "figure_flag": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.0,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)
    assert_allclose(
        [
            float(output["solver/q_R_output"][-1]),
            float(output["solver/q_Z_output"][-1]),
            float(output["solver/q_zeta_output"][-1]),
        ],
        [1.8373067418891855, -0.15175779104290016, -0.0004711781281609329],
        rtol=1e-2,
        atol=1e-2,
    )


def test_sec_harm_3_rel(tmp_path):
    kwargs_dict = {
        "geometry": "cylindrical",
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
        "relativistic_flag": True,
        "temperature_fit_parameters": np.array([10.4]),  # high temperature >5 keV
        "temperature_fit_method": "quadratic",
        "launch_flag": "vacuum_analytic",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.0,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)
    assert_allclose(
        [
            float(output["solver/q_R_output"][-1]),
            float(output["solver/q_Z_output"][-1]),
            float(output["solver/q_zeta_output"][-1]),
        ],
        [1.7142201582176726, -0.19048766599900008, 0.03326574032585646],
        rtol=1e-2,
        atol=1e-2,
    )


def test_sec_harm_4_rel(tmp_path):
    kwargs_dict = {
        "geometry": "cylindrical",
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
        "launch_flag": "vacuum_analytic",
        "Psi_BC_flag": "continuous",
        "relativistic_flag": True,
        "temperature_fit_parameters": np.array([2.8]),  # Low temperature <5 keV
        "temperature_fit_method": "quadratic",
        "figure_flag": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.0,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)
    assert_allclose(
        [
            float(output["solver/q_R_output"][-1]),
            float(output["solver/q_Z_output"][-1]),
            float(output["solver/q_zeta_output"][-1]),
        ],
        [1.393558370046402, -0.3443703253176513, -2.9533975261126915],
        rtol=1e-2,
        atol=1e-2,
    )


def test_sec_harm_5_rel(tmp_path):
    kwargs_dict = {
        "geometry": "cylindrical",
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
        "launch_flag": "vacuum_analytic",
        "Psi_BC_flag": "continuous",
        "relativistic_flag": True,
        "temperature_fit_parameters": np.array([12.9]),  # high temperature >5 keV
        "temperature_fit_method": "quadratic",
        "figure_flag": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.5,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
        "len_tau": 1002,  # needs more points for more accurate determination
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)
    assert_allclose(
        [
            float(output["solver/q_R_output"][-1]),
            float(output["solver/q_Z_output"][-1]),
            float(output["solver/q_zeta_output"][-1]),
        ],
        [1.7165072858256694, -0.03699963289704433, 3.1427380476219575],
        rtol=1e-2,
        atol=1e-2,
    )


def test_sec_harm_6_rel(tmp_path):
    kwargs_dict = {
        "geometry": "cylindrical",
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
        "launch_flag": "vacuum_analytic",
        "Psi_BC_flag": "continuous",
        "relativistic_flag": True,
        "temperature_fit_parameters": np.array([15.3]),  # high temperature >5 keV
        "temperature_fit_method": "quadratic",
        "figure_flag": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.0,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)
    assert_allclose(
        [
            float(output["solver/q_R_output"][-1]),
            float(output["solver/q_Z_output"][-1]),
            float(output["solver/q_zeta_output"][-1]),
        ],
        [1.8134002015544437, -0.16923651161742756, 0.0012300719829534358],
        rtol=1e-2,
        atol=1e-2,
    )


# The resonance flags: first_harmonic_flag, second_harmonic_flag, upper_hybrid_flag


def default_flags_kwargs():
    """test_resonance.py's test_fund_1 ray, with the default flags (all "stop")"""
    return {
        "geometry": "cylindrical",
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
        "launch_flag": "vacuum_analytic",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.5,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }


def test_default_flags_stop(tmp_path):
    kwargs_dict = default_flags_kwargs()
    kwargs_dict["output_path"] = tmp_path
    output = beam_me_up(**kwargs_dict)
    assert_allclose(
        [
            float(output["solver/q_R_output"][-1]),
            float(output["solver/q_Z_output"][-1]),
            float(output["solver/q_zeta_output"][-1]),
        ],
        [1.9162396978780787, -0.05765344712076549, -7.654831340237264e-05],
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("rtol", [1e-3, 1e-4, 1e-6])
def test_output_stops_short_of_the_fundamental(tmp_path, rtol):
    """Within about 3e-5 of the fundamental, K can jump off the dispersion relation
    in one output step, with |H| up to 0.48 and |K| 23% off"""
    kwargs_dict = default_flags_kwargs()
    kwargs_dict.update(output_path=tmp_path, rtol=rtol)
    output = beam_me_up(**kwargs_dict)
    Y = find_normalised_gyro_ang_freq(
        freq_GHz_to_angular_frequency(kwargs_dict["launch_freq_GHz"]),
        output["beam"]["B_magnitude"].values,
    )
    H = np.abs(output["beam"]["H_Booker"].values)
    assert 1e-3 < 1 - Y[-1] < 2e-3
    assert H.max() < 1e-4


def first_harmonic_raise_kwargs(rtol):
    """As `default_flags_kwargs`, but raising at the first harmonic, passing the
    second harmonic, and with coarser output"""
    return {
        "geometry": "cylindrical",
        "poloidal_launch_angle_Torbeam": 3,
        "toroidal_launch_angle_Torbeam": 0,
        "launch_freq_GHz": 33,
        "mode_flag": 1,
        "launch_beam_width": 0.04,
        "launch_beam_curvature": -0.25,
        "launch_position": np.array([3, 0, 0]),
        "density_fit_parameters": np.array([1.0]),
        "rtol": rtol,
        "atol": 1e-6,
        "len_tau": 102,
        "first_harmonic_flag": "raise",
        "second_harmonic_flag": "pass",
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "launch_flag": "vacuum_analytic",
        "Psi_BC_flag": "continuous",
        "figure_flag": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.5,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }


@pytest.mark.parametrize("rtol", [1e-3, 1e-5])
def test_first_harmonic_raise(tmp_path, rtol):
    # The ray isn't stopped at the second harmonic, but it can't go past f_ce = f: there the root selected
    # by mode_flag switches from O to X. Depending on the tolerances, the ray then stays at H = -1 or K
    # jumps onto the other root with H near 0 (beam-tracing/Scotty#169), so it raises at either.
    # The error points to first_harmonic_flag="stop"
    kwargs_dict = first_harmonic_raise_kwargs(rtol)
    kwargs_dict["output_path"] = tmp_path
    with pytest.raises(RuntimeError, match="reached the first harmonic") as error:
        beam_me_up(**kwargs_dict)
    assert '`first_harmonic_flag="stop"`' in str(error.value)


@pytest.mark.parametrize("rtol", [1e-3, 1e-5])
def test_first_harmonic_stop_second_pass(tmp_path, rtol):
    """The same ray stopping at the first harmonic and passing the second is traced
    up to 1e-3 short of the fundamental and returned, as for a two-stage run that
    restarts past the layer"""
    kwargs_dict = first_harmonic_raise_kwargs(rtol)
    kwargs_dict.update(
        output_path=tmp_path, first_harmonic_flag="stop", second_harmonic_flag="pass"
    )
    output = beam_me_up(**kwargs_dict)
    Y = find_normalised_gyro_ang_freq(
        freq_GHz_to_angular_frequency(kwargs_dict["launch_freq_GHz"]),
        output["beam"]["B_magnitude"].values,
    )
    assert 1e-3 < 1 - Y[-1] < 3e-3
    assert np.abs(output["beam"]["H_Booker"].values).max() < 1e-2
