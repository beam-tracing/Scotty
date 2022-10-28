from scotty.init_bruv import ne_settings, beam_settings

import numpy as np
import numpy.testing as nt
import pytest


def test_ne_settings():
    # Odd sort of golden answer test: check we can recover known values
    fit_param, fit_time, poloidal_flux_enter = ne_settings(
        "DBS_NSTX_MAST", 29908, 0.170, "poly3"
    )
    assert fit_time == 0.170
    nt.assert_array_equal(fit_param, [-3.31670147, 2.24970438, -0.46971473, 2.47113803])
    assert poloidal_flux_enter == 1.13336773


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
    width, curvature = beam_settings(diagnostic, method, launch_freq_GHz)
    assert np.isclose(width, expected_width)
    assert np.isclose(curvature, expected_curvature)
