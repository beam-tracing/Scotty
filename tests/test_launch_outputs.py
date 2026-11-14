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
    kwargs_dict["len_tau"] = 10
    kwargs_dict["output_path"] = tmp_path_factory.mktemp("launch_outputs")
    return request.param, beam_me_up(**kwargs_dict)


def test_K_launch_is_the_launch_direction(steered_run):
    """K_launch_cartesian is |K| = K0 along the TORBEAM launch direction"""
    toroidal_angle, dt = steered_run
    K_cart = dt["solver/K_launch_cartesian"].values
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
    so = dt["solver"]

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
    so = dt["solver"]
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
    so = dt["solver"]

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


def test_Psi_entry_cartesian_is_the_launched_beam_at_the_entry(steered_run):
    """The saved Cartesian entry Psi is the launched Psi propagated through
    vacuum to the entry, Psi_w^-1 -> Psi_w^-1 + d / K0, with no component
    along K. The launched Psi_w is the circular Gaussian of the inputs, the
    same in any transverse basis. The cylindrical entry Psi has terms in K_R
    and K_zeta at the entry, which differ from those at the launch point once
    the beam is steered toroidally"""
    _, dt = steered_run
    so = dt["solver"]

    K = so["K_launch_cartesian"].values
    g_hat = K / np.linalg.norm(K)
    a_hat = np.cross(g_hat, [0.0, 0.0, 1.0])
    a_hat /= np.linalg.norm(a_hat)
    basis = np.array([a_hat, np.cross(g_hat, a_hat)])
    inputs = dt["inputs"]
    K0 = float(inputs["launch_wavenumber"])
    width = float(inputs["launch_beam_width"])
    curvature = float(inputs["launch_beam_curvature"])
    distance = float(so["distance_from_launch_to_entry"])

    Psi_w_launch = (K0 * curvature + 2j / width**2) * np.eye(2)
    expected = np.linalg.inv(np.linalg.inv(Psi_w_launch) + distance / K0 * np.eye(2))
    Psi_entry = so["Psi_3D_entry_labframe_cartesian"].values

    scale = np.abs(expected).max()
    assert_allclose(basis @ Psi_entry @ basis.T, expected, rtol=0, atol=1e-9 * scale)
    assert_allclose(Psi_entry @ g_hat, 0, atol=1e-9 * scale)
