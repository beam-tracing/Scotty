"""Checks of the ray piece of the localisation, loc_r = g_ant^2 / g^2, and of the
group velocity g = grad_K H it uses (H the eigenvalue of D, from Cardano's formula).

Both checks need a toroidally steered beam: at a toroidal launch angle of zero,
the toroidal component of g vanishes, and so does any error in it.
"""

from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty

import numpy as np
import pytest
from scipy import constants


@pytest.fixture(scope="module", params=[(0.0, 1), (10.0, 1), (10.0, -1)], ids=["O-tor0", "O-tor10", "X-tor10"])
def steered_run(request, tmp_path_factory):
    toroidal_angle, mode_flag = request.param
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    kwargs_dict["find_B_method"] = "unit-tests"
    kwargs_dict["toroidal_launch_angle_Torbeam"] = toroidal_angle
    kwargs_dict["mode_flag"] = mode_flag
    kwargs_dict["figure_flag"] = False
    kwargs_dict["output_path"] = tmp_path_factory.mktemp("localisation")
    return beam_me_up(**kwargs_dict)


def test_loc_r_vacuum_limit(steered_run):
    """The beam enters the plasma where the density is zero, so the first point
    is in vacuum, where g = g_ant = 2c/omega and loc_r = 1"""
    analysis = steered_run["analysis"]
    omega = float(steered_run["inputs"]["launch_angular_frequency"])
    g_first = float(np.real(analysis["g_magnitude_Cardano"].values[0]))

    assert np.isclose(g_first * omega / (2 * constants.c), 1.0, rtol=1e-5, atol=0)
    assert np.isclose(float(np.real(analysis["loc_r"].values[0])), 1.0, rtol=1e-5, atol=0)


def test_g_Cardano_parallel_to_g_Booker(steered_run):
    """The Booker dispersion relation and the eigenvalue of D both vanish on
    the ray, so their K-gradients must be parallel there (eq. 46 of Hall-Chen,
    Parra & Hillesheim, PPCF 64 095002, 2022). This fails if the toroidal
    component has the wrong units, or if theta_m is held fixed while K is
    varied (which makes g parallel to K instead)"""
    analysis = steered_run["analysis"]
    cos_angle = np.abs(np.sum(analysis["g_hat_Cardano"].values * analysis["g_hat"].values, axis=-1))
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    assert angle_deg.max() < 0.1
