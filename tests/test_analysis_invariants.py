"""Invariants of the analysis on a toroidally steered beam.

The golden tests launch at a toroidal angle of zero, where the beam stays near
zeta = 0 and errors in the cylindrical-to-Cartesian conversion of the field
vanish. These cases steer the beam toroidally so that zeta changes along the ray.
"""

from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty

import numpy as np
import pytest


@pytest.fixture(
    scope="module",
    params=[(0.0, 1), (10.0, 1), (10.0, -1)],
    ids=["O-tor0", "O-tor10", "X-tor10"],
)
def steered_run(request, tmp_path_factory):
    toroidal_angle, mode_flag = request.param
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    kwargs_dict["toroidal_launch_angle_Torbeam"] = toroidal_angle
    kwargs_dict["mode_flag"] = mode_flag
    kwargs_dict["figure_flag"] = False
    kwargs_dict["output_path"] = tmp_path_factory.mktemp("analysis_invariants")
    return beam_me_up(**kwargs_dict)


def test_b_hat_matches_field_components(steered_run):
    """b_hat_cartesian is the unit vector of the stored (B_X, B_Y, B_Z)"""
    analysis = steered_run["analysis"]
    B = np.stack(
        [analysis["B_X"].values, analysis["B_Y"].values, analysis["B_Z"].values],
        axis=-1,
    )
    np.testing.assert_allclose(
        analysis["b_hat_cartesian"].values,
        B / np.linalg.norm(B, axis=-1, keepdims=True),
        rtol=0,
        atol=1e-10,
    )


def test_polarisation_matches_mode(steered_run, request):
    """At the cut-off, O-mode is polarised mostly along b and X-mode mostly
    across it. e_hat is an eigenvector of D, so it is in the (e_1, e_2, b)
    basis of eqs. 36-41 (despite its "col_cart" label) and the b-component
    is its third entry. Fails if e_hat is taken from the wrong eigenvector
    (the eigh ordering is not the Cardano ordering)"""
    analysis = steered_run["analysis"]
    i = int(analysis["cutoff_index"])
    e_hat = analysis["e_hat"].values[i]
    e_b = np.abs(e_hat[2]) / np.linalg.norm(e_hat)

    if "X-" in request.node.callspec.id:
        assert e_b < 0.2
    else:
        assert e_b > 0.7


def test_eigenvalue_of_D_consistent_with_Booker(steered_run):
    """The Booker H and the eigenvalue of D (Cardano) vanish on the same
    surface, so on the ray the Cardano H can be no larger than the solver's
    drift off that surface. An inconsistent b_hat between the Hamiltonian and
    the analysis breaks this"""
    analysis = steered_run["analysis"]
    H_Cardano = np.real(analysis["H_Cardano_solution"].values)
    H_Booker = np.real(analysis["H_Booker"].values)

    assert np.max(np.abs(H_Cardano)) < 2 * np.max(np.abs(H_Booker)) + 1e-6


def test_dispersion_relation_satisfied_on_ray(steered_run, request):
    """The ray is traced on H = 0, so the Booker H must stay near zero"""
    if "X-tor10" in request.node.callspec.id:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="X-mode at 10 deg: |H_Booker| reaches ~4e-3 in the last few "
                "points at the default rtol 1e-3 (5e-7 at rtol 1e-5): solver step "
                "error on K ~ 1000 m^-1, not a Hamiltonian error. A tolerance-aware "
                "check should replace this xfail",
            )
        )
    H_Booker = np.real(steered_run["analysis"]["H_Booker"].values)

    assert np.max(np.abs(H_Booker)) < 1e-3
