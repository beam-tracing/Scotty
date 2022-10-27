from scotty.init_bruv import ne_settings

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
