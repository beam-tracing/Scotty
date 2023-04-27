from scotty.init_bruv import ne_settings, beam_settings, get_parameters_for_Scotty

import numpy as np
import numpy.testing as nt
import pytest


def test_ne_settings():
    # Odd sort of golden answer test: check we can recover known values
    fit_param, fit_time, poloidal_flux_zero_density = ne_settings(
        "DBS_NSTX_MAST", 29908, 0.170, "poly3"
    )
    assert fit_time == 0.170
    nt.assert_array_equal(
        fit_param.coefficients, [-3.31670147, 2.24970438, -0.46971473, 2.47113803]
    )
    assert poloidal_flux_zero_density == 1.13336773


def test_bad_ne_settings():
    # Check we get an error for known non-values
    with pytest.raises(ValueError):
        ne_settings("NOT A DIAGNOSTIC", 29908, 0.170, "tanh")

    with pytest.raises(ValueError):
        ne_settings("DBS_NSTX_MAST", 3.14, 0.170, "tanh")

    with pytest.raises(ValueError):
        ne_settings("DBS_NSTX_MAST", 29908, 0.170, "unknown method")


@pytest.mark.parametrize(
    ("diagnostic", "method", "launch_freq_GHz", "expected_width", "expected_curvature"),
    [
        ("DBS_NSTX_MAST", "data", 35.0, 0.04303, -0.2608064061039988),
        ("DBS_CIEMAT_JT60SA", "data", 90.0, 0.06323503329291348, -0.5535179506038995),
        ("DBS_UCLA_DIII-D_240", "thin_lens", 55.0, 0.02216858596, -1.0831303),
        ("DBS_UCLA_MAST-U", "thin_lens", 48.0, 0.05165271, -0.7997469),
        # TODO: Check following, have different signs and magnitude to others
        ("DBS_SWIP_MAST-U", "estimate_var_w0", 52.0, -11.518604, 26.2085140),
        ("DBS_SWIP_MAST-U", "estimate_fix_w0", 52.0, -12.352420, 24.4393811),
    ],
)
def test_beam_settings(
    diagnostic, method, launch_freq_GHz, expected_width, expected_curvature
):
    """Golden answer test for various diagnostics"""
    width, curvature = beam_settings(diagnostic, launch_freq_GHz, method)
    assert np.isclose(width, expected_width)
    assert np.isclose(curvature, expected_curvature)


def test_bad_beam_settings():
    # Check we get an error for known non-values
    with pytest.raises(ValueError):
        beam_settings("NOT A DIAGNOSTIC", 52.0, "data")

    with pytest.raises(ValueError):
        beam_settings("DBS_NSTX_MAST", 52.0, "unknown method")


def test_parameters_DBS_NSTX_MAST():
    """Golden answer test"""

    parameters = get_parameters_for_Scotty(
        "DBS_NSTX_MAST", launch_freq_GHz=52.0, mirror_rotation=2, mirror_tilt=4
    )

    expected = {
        "poloidal_launch_angle_Torbeam": -5.446475141297062,
        "toroidal_launch_angle_Torbeam": 1.3370817251108178,
        "launch_freq_GHz": 52,
        "mode_flag": None,
        "launch_beam_width": 0.03629814335,
        "launch_beam_curvature": -0.7714837802263125,
        "launch_position": np.array([2.43521, 0.0, 0.0]),
        "Psi_BC_flag": "continuous",
        "figure_flag": True,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
    }
    assert sorted(expected.keys()) == sorted(parameters.keys())
    for key, value in expected.items():
        if isinstance(value, (bool, type(None), str)):
            assert parameters[key] == value, key
        else:
            assert np.allclose(parameters[key], value), key


def test_parameters_DBS_UCLA_DIII_D_240():
    """Golden answer test"""

    parameters = get_parameters_for_Scotty("DBS_UCLA_DIII-D_240", launch_freq_GHz=52.0)

    expected = {
        "poloidal_launch_angle_Torbeam": None,
        "toroidal_launch_angle_Torbeam": None,
        "launch_freq_GHz": 52,
        "mode_flag": None,
        "launch_beam_width": 0.02173742019305591,
        "launch_beam_curvature": -1.2659055507320647,
        "launch_position": np.array([2.587, 0.0, -0.0157]),
        "Psi_BC_flag": "continuous",
        "figure_flag": True,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
    }
    assert sorted(expected.keys()) == sorted(parameters.keys())
    for key, value in expected.items():
        print(key, value, parameters[key])
        if isinstance(value, (bool, type(None), str)):
            assert parameters[key] == value, key
        else:
            assert np.allclose(parameters[key], value), key
