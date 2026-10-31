"""Checks of the launch values saved in ``solver_output``, for launches with
a toroidal angle, where cylindrical and Cartesian components differ.

In cylindrical coordinates K_zeta is the toroidal mode number, q_R times the
Cartesian K_Y at zeta = 0, and it is conserved along the ray because the field
is axisymmetric.
"""

from scotty.beam_me_up import beam_me_up
from scotty.fun_general import (
    find_K_labframe_cart_to_cyl,
    find_Psi_3D_labframe_cyl_to_cart,
)
from scotty.init_bruv import get_parameters_for_Scotty

import numpy as np
from numpy.testing import assert_allclose
import pytest


@pytest.fixture(
    scope="module",
    params=[0.0, 10.0, -15.0],
    ids=["tor0", "tor10", "tor-15"],
)
def steered_run(request, tmp_path_factory):
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    kwargs_dict["toroidal_launch_angle_Torbeam"] = request.param
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["output_path"] = tmp_path_factory.mktemp("launch_outputs")
    return request.param, beam_me_up(**kwargs_dict)


def test_K_launch_is_the_launch_direction(steered_run):
    """K_launch_cartesian is |K| = K0 along the TORBEAM launch direction"""
    toroidal_angle, dt = steered_run
    K_cart = dt["solver_output/K_launch_cartesian"].values
    K0 = float(dt["inputs/launch_wavenumber"])
    pol = np.deg2rad(float(dt["inputs/poloidal_launch_angle_Torbeam"]))
    tor = np.deg2rad(toroidal_angle)

    expected = -K0 * np.array(
        [np.cos(pol) * np.cos(tor), np.cos(pol) * np.sin(tor), np.sin(pol)]
    )
    assert_allclose(K_cart, expected, rtol=1e-12, atol=1e-9)


def test_K_launch_cylindrical_matches_cartesian(steered_run):
    """The two saved launch wavevectors are the same vector"""
    _, dt = steered_run
    so = dt["solver_output"]

    assert_allclose(
        so["K_launch_cylindrical"].values,
        find_K_labframe_cart_to_cyl(
            so["K_launch_cartesian"].values, so["q_launch_cartesian"].values
        ),
        rtol=1e-12,
        atol=1e-9,
    )


def test_K_zeta_launch_is_the_conserved_mode_number(steered_run):
    """K_zeta at launch is the toroidal mode number the beam keeps along the
    ray, as in v3 (K_zeta_launch = -K0 sin(tor) cos(pol) R_launch)"""
    toroidal_angle, dt = steered_run
    so = dt["solver_output"]
    K0 = float(dt["inputs/launch_wavenumber"])
    pol = np.deg2rad(float(dt["inputs/poloidal_launch_angle_Torbeam"]))
    R_launch = float(so["q_launch_cylindrical"].values[0])

    K_zeta_launch = float(so["K_launch_cylindrical"].values[1])
    v3_K_zeta_launch = -K0 * np.sin(np.deg2rad(toroidal_angle)) * np.cos(pol) * R_launch

    assert np.isclose(K_zeta_launch, v3_K_zeta_launch, rtol=1e-12, atol=1e-9)
    assert_allclose(so["K_zeta_output"].values, K_zeta_launch, rtol=1e-9, atol=1e-9)


def test_Psi_launch_cylindrical_matches_cartesian(steered_run):
    """The saved launch Psi in each geometry is the same beam"""
    _, dt = steered_run
    so = dt["solver_output"]

    assert_allclose(
        find_Psi_3D_labframe_cyl_to_cart(
            so["Psi_3D_launch_labframe_cylindrical"].values,
            so["K_launch_cylindrical"].values,
            so["q_launch_cylindrical"].values,
        ),
        so["Psi_3D_launch_labframe_cartesian"].values,
        rtol=1e-9,
        atol=1e-9,
    )
